import os, sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from transformers import DecisionTransformerConfig

from DT import OHLCV_Transformer
import decision_transformer.manage_time as ART_manager
from decision_transformer.manage_time import device

# Initial parameters
model_name_4_saving = 'checkpoint_ff_time_40_100_random'
mdp_constr = True
datasets, dataloaders = ART_manager.get_train_val_test_data(mdp_constr=mdp_constr, timestep_norm=False)
train_loader, eval_loader, test_loader = dataloaders
n_state = train_loader.dataset.n_state
n_data = train_loader.dataset.n_data
n_time = train_loader.dataset.max_len

# Transformer parameters
config = DecisionTransformerConfig(
    state_dim=n_state, 
    d_model=384,
    max_window=n_time,
    vocab_size=1,
    n_positions=1024,
    n_layer=6,
    n_head=6,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    )
model = OHLCV_Transformer(config)  # AutonomousFreeflyerTransformer_pred_time(config) 
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")
model.to(device);

# training
optimizer = AdamW(model.parameters(), lr=3e-5)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = 1000000000 # train until convergence

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

# To activate only when starting from a pretrained model
# accelerator.load_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_dataset)


# evaluate the model on validation data during training every eval_steps
@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    losses_state = []
    losses_return = []
    eval_iters = 100
    for step in range(eval_iters):
        data_iter = iter(eval_dataloader)
        time_i, state_i, volume_i, return_i, attention_mask_i, _, _, _ = next(data_iter)
        with torch.no_grad():
            # def forward(
            # self,
            # states: Optional[torch.FloatTensor] = None,
            # returns: Optional[torch.FloatTensor] = None,
            # volumes: Optional[torch.FloatTensor] = None,
            # timestamps: Optional[torch.FloatTensor] = None,
            # hours: Optional[torch.FloatTensor] = None,
            # timesteps: Optional[torch.LongTensor] = None,
            # attention_mask: Optional[torch.FloatTensor] = None,
            # output_hidden_states: Optional[bool] = None,
            # output_attentions: Optional[bool] = None,
            # return_dict: Optional[bool] = None,
            state_preds, return_preds = model(
                states=states_i,
                volumes=volume_i,
                returns=returns_i,
                timestamps=timestamps_i,
                timesteps=timesteps_i,
                hours=hours_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )

        alpha = 0.5  # small weight for state prediction
        loss_i_return = torch.mean((return_preds - returns_i) ** 2)
        loss_i_state = torch.mean((state_preds[:, :-1, :] - states_i[:, 1:, :]) ** 2)
        loss_i = loss_i_return + alpha * loss_i_state

        losses.append(accelerator.gather(loss_i))
        losses_return.append(accelerator.gather(loss_i_return))
        losses_state.append(accelerator.gather(loss_i_state))

    
    loss = torch.mean(torch.tensor(losses))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss_return = torch.mean(torch.tensor(losses_return))
    model.train()
    return loss.item(), loss_state.item(), loss_return.item()



# Ensures the evaluation loop is working.
# Confirms the data pipeline (eval_dataloader) and model forward pass are behaving correctly.
# If something crashes here, you know it's not due to the training loop.
eval_loss, loss_state, loss_return = evaluate()
accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_return})

# Training
eval_steps = 500 # After every 500 training steps, evaluate and log the performance.
samples_per_step = accelerator.state.num_processes * train_loader.batch_size # logging how many training samples have been seen
#torch.manual_seed(4)

model.train()
completed_steps = 0
log = {
    'loss':[],
    'loss_state':[],
    'loss_return':[]
}
'''log = np.load(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_saving + '/log.npz', allow_pickle=True)['log'].item()'''
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        with accelerator.accumulate(model):
            time_i, state_i, volume_i, return_i, attention_mask_i, _, _, _ = batch
            state_preds, return_preds = model(
                states=states_i,
                volumes=volume_i,
                returns=returns_i,
                timestamps=timestamps_i,
                timesteps=timesteps_i,
                hours=hours_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
            loss_i_return = torch.mean((return_preds - returns_i) ** 2)
            loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
            loss = loss_i_return + loss_i_state
            if step % 100 == 0:
                accelerator.print(
                    {
                        "lr": lr_scheduler.get_lr(),
                        "samples": step * samples_per_step,
                        "steps": completed_steps,
                        "loss/train": loss.item(),
                    }
                )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            if (step % (eval_steps)) == 0:
                eval_loss, loss_state, loss_return = evaluate()
                accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_return})
                log['loss'].append(eval_loss)
                log['loss_state'].append(loss_state)
                log['loss_return'].append(loss_return)
                model.train()
                accelerator.wait_for_everyone()
            if (step % (eval_steps*10)) == 0: # save updates every 10*eval_steps=5000 steps
                print('Saving model..')
                accelerator.save_state(root_folder+'/decision_transformer/saved_files/checkpoints/'+model_name_4_saving)
                np.savez_compressed(root_folder + '/decision_transformer/saved_files/checkpoints/' +model_name_4_saving+ '/log',
                            log = log
                            )