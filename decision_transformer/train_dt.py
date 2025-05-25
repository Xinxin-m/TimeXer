import os
import sys
import numpy as np
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from transformers import DecisionTransformerConfig

from DT import OHLCV_Transformer
from decision_transformer.data_factory_dt import data_provider_dt

class Args:
    def __init__(self):
        self.root_path = './data/'  # Update this to your data path
        self.batch_size = 32
        self.num_workers = 0
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24

def train():
    # Initialize arguments
    args = Args()
    
    # Get dataloaders
    train_dataset, train_loader = data_provider_dt(args, flag='train')
    val_dataset, val_loader = data_provider_dt(args, flag='val')
    test_dataset, test_loader = data_provider_dt(args, flag='test')
    
    # Get dimensions from dataset
    n_state = train_dataset.state_dim
    n_data = len(train_dataset)
    n_time = args.seq_len
    
    # Model configuration
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
    
    # Initialize model
    model = OHLCV_Transformer(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Learning rate scheduler
    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = 1000000000  # train until convergence
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps,
    )
    
    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        losses_state = []
        losses_return = []
        eval_iters = 100
        
        for step in range(eval_iters):
            data_iter = iter(eval_dataloader)
            batch = next(data_iter)
            timestamp, state, volume, returns, attention_mask, timestep, _, _ = batch
            
            state_preds, return_preds = model(
                states=state,
                volumes=volume,
                returns=returns,
                timestamps=timestamp,
                timesteps=timestep,
                hours=torch.zeros_like(timestamp),  # Not used
                attention_mask=attention_mask,
                return_dict=False,
            )
            
            alpha = 0.5  # small weight for state prediction
            loss_i_return = torch.mean((return_preds - returns) ** 2)
            loss_i_state = torch.mean((state_preds[:, :-1, :] - state[:, 1:, :]) ** 2)
            loss_i = loss_i_return + alpha * loss_i_state
            
            losses.append(accelerator.gather(loss_i))
            losses_return.append(accelerator.gather(loss_i_return))
            losses_state.append(accelerator.gather(loss_i_state))
        
        loss = torch.mean(torch.tensor(losses))
        loss_state = torch.mean(torch.tensor(losses_state))
        loss_return = torch.mean(torch.tensor(losses_return))
        model.train()
        return loss.item(), loss_state.item(), loss_return.item()
    
    # Training loop
    eval_steps = 500
    samples_per_step = accelerator.state.num_processes * train_loader.batch_size
    
    model.train()
    completed_steps = 0
    log = {
        'loss': [],
        'loss_state': [],
        'loss_return': []
    }
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader, start=0):
            with accelerator.accumulate(model):
                timestamp, state, volume, returns, attention_mask, timestep, _, _ = batch
                
                state_preds, return_preds = model(
                    states=state,
                    volumes=volume,
                    returns=returns,
                    timestamps=timestamp,
                    timesteps=timestep,
                    hours=torch.zeros_like(timestamp),  # Not used
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                
                loss_i_return = torch.mean((return_preds - returns) ** 2)
                loss_i_state = torch.mean((state_preds[:, :-1, :] - state[:, 1:, :]) ** 2)
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
                
                if (step % eval_steps) == 0:
                    eval_loss, loss_state, loss_return = evaluate()
                    accelerator.print({
                        "loss/eval": eval_loss,
                        "loss/state": loss_state,
                        "loss/action": loss_return
                    })
                    log['loss'].append(eval_loss)
                    log['loss_state'].append(loss_state)
                    log['loss_return'].append(loss_return)
                    model.train()
                    accelerator.wait_for_everyone()
                
                if (step % (eval_steps*10)) == 0:
                    print('Saving model..')
                    accelerator.save_state('./checkpoints/model')
                    np.savez_compressed('./checkpoints/log.npz', log=log)

if __name__ == "__main__":
    train() 