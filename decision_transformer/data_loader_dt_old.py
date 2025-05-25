import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class OHLCV_Dataset(Dataset):
    """
    Dataset class for Decision Transformer format data.
    
    Data contains:
    - timestamp: int64 timestamp for each data point
    - state: ohlcv values 
    - return: log_return 

    data_x, data_y contain the same data columns but y is shifted (y is the the data over the target time window for prediction)
    """
    def __init__(self, rootpath, filename, timestep=3600, flag='train', size=None, scale=True):
        # size [seq_len, label_len, pred_len]
        # label_len is the length of the overlap between the input and target data
        self.rootpath = rootpath
        self.filename = filename
        self.timestep = timestep
        self.flag = flag
        self.scale = scale # boolean to scale the data or not
  
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag] # 0,1,2

        # Set sequence lengths
        if size is None:
            self.seq_len = 720
            self.label_len = 120
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.__read_data__()


    def __read_data__(self):
        filepath = os.path.join(self.rootpath, self.filename)

        if self.filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Ensure log_return (value to be predicted) is the last column
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'log_return']
        if 'hour' in df.columns:
            required_columns.insert(1, 'hour') # put hour in the second column
        df = df[required_columns]

        # Drop rows with any NaN values
        original_len = len(df)
        df = df.dropna()
        dropped_rows = original_len - len(df)
        if dropped_rows > 0:
            print(f"\nDropped {dropped_rows} rows ({dropped_rows/original_len*100:.2f}%) containing NaN values")

        # Split data into train/val/test
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_val = len(df) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]
        border1 = border1s[self.set_type] # select the train/val/test chunk of data
        border2 = border2s[self.set_type]
        
        # Extract features
        self.timestamps = df['timestamp'].values[border1:border2]
        df = df.drop('timestamp', axis=1)

        # Define data (np.array of all variables except time)
        # Scale the data if needed (standardize each column)
        if self.scale:
            # scale wrt mean and std of training data
            train_data = df[border1s[0]:border2s[0]]
            self.scaler = StandardScaler() # from sklearn (normalize across columns of the df)
            self.scaler.fit(train_data.values)
            # df.values is a np.array of shape (n_samples, n_features)
            data = self.scaler.transform(df.values)
        else:
            data = df.values    # ['hours', 'open', 'high', 'low', 'close', 'volume', 'log_return'] in this order
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
  
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        
        # Get input sequence
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = (self.timestamps[s_begin:s_end] - self.timestamps[s_begin]) // self.timestep
        seq_y_mark = (self.timestamps[r_begin:r_end] - self.timestamps[s_begin]) // self.timestep
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self): # number of data samples created = num subsequences of (seq_len+pred_len) in data[border1:border2]
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data): # inverse the scaling
        return self.scaler.inverse_transform(data) 


class CombinedDataset(Dataset):
    """
    Dataset class that reads from a list of data files.
    Each file is loaded separately using OHLCV_Dataset and samples are drawn from all files.
    
    Args:
        rootpath (str): Root directory containing all stock data files
        filenames (list): List of filenames to load
        flag (str): 'train', 'val', or 'test'
        size (list): [seq_len, label_len, pred_len]
        scale (bool): Whether to scale the data
    """
    def __init__(self, rootpath, filenames, flag='train', size=None, scale=True):
        self.rootpath = rootpath
        self.filenames = filenames
        self.flag = flag
        self.size = size
        self.scale = scale
        
        # Create individual datasets for each file
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]  # For indexing across all datasets
        
        for filename in filenames:
            dataset = OHLCV_Dataset(
                rootpath=rootpath,
                filename=filename,
                flag=flag,
                size=size,
                scale=scale
            )
            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
            
        self.total_length = sum(self.dataset_lengths)
        
    def __getitem__(self, index):
        # Find which dataset this index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1 and index >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
            
        # Get the local index within the dataset
        local_index = index - self.cumulative_lengths[dataset_idx]
        
        # Return the item from the appropriate dataset
        return self.datasets[dataset_idx][local_index]
    
    def __len__(self):
        return self.total_length
    
    def inverse_transform(self, data):
        # Use the scaler from the first dataset for inverse transform
        # This assumes all datasets use the same scaling
        return self.datasets[0].inverse_transform(data) 