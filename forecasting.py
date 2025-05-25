import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from decision_transformer.data_factory_dt import get_dataloader
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models.TimeXer_time import Model

class OHLCV_Forecast:
    """
    A class for training and evaluating OHLCV (Open, High, Low, Close, Volume) forecasting models.
    
    This class supports both single dataset and multiple datasets training.
    
    Required Parameters (args):
        - rootpath: str, Path to the data directory
        - filename: str, Name of the data file (for single dataset mode)
        - filenames: list, List of filenames (for multiple datasets mode)
        - timestep: int, Timestep for the data
        - batch_size: int, Batch size for training
        - seq_len: int, Input sequence length
        - label_len: int, Label sequence length
        - pred_len: int, Prediction sequence length
        - use_datetime: bool, Whether to use datetime features 
        - perc_missing: float, Percentage of missing values allowed in a sequence
        - chunksize: int, Maximum size of consecutive missing chunks allowed
        - colnames: list, List of column names to use (default ohlcvr)
        - normalize: bool, Whether to normalize data (apart from timestamp column)
        - num_workers: int, Number of workers for data loading
        - learning_rate: float, Learning rate for training
        - train_epochs: int, Number of training epochs
        - patience: int, Patience for early stopping
        - use_gpu: bool, Whether to use GPU
        - gpu: int, GPU device ID if use_gpu is True
        - checkpoints: str, Path to save model checkpoints
        - inverse: bool, Whether to inverse transform predictions
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = Model(args).float().to(self.device)
        
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        dataset, dataloader = get_dataloader(self.args, flag)
        return dataset, dataloader

    def evaluate(self, val_data, val_loader):
        total_loss = []
        Loss = nn.MSELoss() # initiate the class
        self.model.eval()
        with torch.no_grad():
            for batch_indices, batch_data in val_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark)
                # Assume returns are the last column of the output
                pred_returns = outputs[:, -self.args.pred_len:, -1]
                true_returns = batch_y[:, -self.args.pred_len:, -1]
                pred = pred_returns.detach().cpu()
                true = true_returns.detach().cpu()

                loss = Loss(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        Loss = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for batch_indices, batch_data in train_loader:
                iter_count += 1
                optimizer.zero_grad()
                
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Use last column (log_return pred) to compute loss
                outputs = self.model(batch_x, batch_x_mark)
                outputs = outputs[:, -self.args.pred_len:, -1]
                batch_y = batch_y[:, -self.args.pred_len:, -1]
                
                loss = Loss(outputs, batch_y) 
                train_loss.append(loss.item())

                if iter_count % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter_count, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / 100  # Calculate speed for last 100 iterations
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_count)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()  # Reset timer for next 100 iterations

                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            val_loss = self.evaluate(val_data, val_loader)
            test_loss = self.evaluate(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss, test_loss))
            
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # The code uses an EarlyStopping class from utils/tools.py that saves the model whenever validation loss improves
            # this becomes the best model
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        """
        Args:
            setting: str, Name of the experiment setting (used in checkpoint savepath)
            test: int, Whether to load a saved model (1) or use the current model (0)
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for batch_indices, batch_data in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)
                outputs = outputs[:, -self.args.pred_len:, :]  # [batch_size, pred_len, n_vars]
                batch_y = batch_y[:, -self.args.pred_len:, :]  # [batch_size, pred_len, n_vars]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse and batch_indices is not None:
                    shape = batch_y.shape
                    # batch_indices: [batch_size]
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1), batch_indices).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1), batch_indices).reshape(shape)

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                
                # For visualization, use the first sample in the batch
                if batch_indices is not None and isinstance(batch_indices, (list, np.ndarray)) and len(preds) % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1), batch_indices).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(len(preds)) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_ohlcv_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return 