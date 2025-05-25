import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

# handle missing values in dataset by selecting chunks with < perc_missing missing values and no chunks with > chunksize missing values
class OHLCV_Dataset(Dataset):
    """
    Dataset class for Decision Transformer format data with continuous sequence handling.
    
    Data contains:
    - timestamp: int64 timestamp for each data point
    - state: ohlcv values 
    - return: log_return 

    data_x, data_y contain the same data columns but y is shifted (y is the the data over the target time window for prediction)
    """
    def __init__(self, rootpath, filename, timestep=3600, flag='train', size=None, scale=True, 
                 perc_missing=5, chunksize=3, colnames=None, use_datetime=True, patch_len=None):
        # size [seq_len, label_len, pred_len]
        # label_len is the length of the overlap between the input and target data
        # colnames: all variables used: ensure it has 'timestamp' column and log_return (value to be predicted) is the last column
        # use_datetime: if True, then use data_stamp with month, day, weekday, hour, else use indices (regularized timestamps)
        self.rootpath = rootpath
        self.filename = filename
        self.timestep = timestep
        self.flag = flag
        self.scale = scale # boolean to scale the data or not
        self.perc_missing = perc_missing
        self.chunksize = chunksize
        self.colnames = colnames if colnames is not None else ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'log_return']
        self.use_datetime = use_datetime
        self.data_stamp = None # same as df_stamp in the original data_loader.py (when timeenc=0) but with index column
        self.use_datetime = use_datetime

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag] # 0,1,2

        # Set sequence lengths
        if size is None:
            self.seq_len = 336
            self.label_len = 72
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.sample_len = self.seq_len + self.pred_len
        self.split_data()


    def get_df(self):
        '''
        Add indices, drop timestamp, dropna, select useful columns
        '''
        filepath = os.path.join(self.rootpath, self.filename)
  
        if self.filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)
        df = df[self.colnames].replace([np.inf, -np.inf], np.nan).dropna()

        if self.use_datetime:
            self.data_stamp = pd.DataFrame()
            self.data_stamp['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # this ordering ensures time is the first column
            self.data_stamp['month'] = self.data_stamp['timestamp'].dt.month
            self.data_stamp['day'] = self.data_stamp['timestamp'].dt.day
            self.data_stamp['weekday'] = self.data_stamp['timestamp'].dt.weekday # 0-6 
            self.data_stamp['hour'] = self.data_stamp['timestamp'].dt.hour # 0-23
            self.data_stamp['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) // self.timestep

        df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) // self.timestep
        df.set_index('timestamp', inplace=True) # this drops the timestamp column 
        return df
 

    def find_valid_start_ind(self, df):
        """
        Find all valid starting indices for sequences that satisfy the missing data criteria.
        Uses vectorized operations for better performance.
        """
        existing_indices = df.index.values.astype(int) # 1D np.array of time indices

        # Create a boolean array for all possible starting indices
        n_sequences = len(df) - self.sample_len + 1 # number of possible starting indices
        valid_sequences = np.ones(n_sequences, dtype=bool)
        
        # Vectorized check for missing values in each sequence
        for i in range(n_sequences):
            seq_indices = existing_indices[i:i + self.sample_len]
            expected_seq = np.arange(seq_indices[0], seq_indices[0] + self.sample_len)
            is_missing = ~np.isin(expected_seq, seq_indices)
            
            # Check percentage of missing values
            missing_perc = (is_missing.sum() / self.sample_len) * 100
            if missing_perc > self.perc_missing:
                valid_sequences[i] = False
                continue
                            # Check for max consecutive missing chunk
            if is_missing.any():
                # Find consecutive missing chunks using diff
                diff = np.diff(np.concatenate(([0], is_missing.astype(int), [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                chunk_sizes = ends - starts
                if np.any(chunk_sizes > self.chunksize):
                    valid_sequences[i] = False
        
        start_rows = np.where(valid_sequences)[0] # row numbers
        start_indices = existing_indices[start_rows] # time indices on those rows
        return start_indices, start_rows
    
    def split_data(self):
        df = self.get_df()
        self.indices = df.index.values # all existing time indices
        # Find valid starting indices
        self.start_indices, self.start_rows = self.find_valid_start_ind(df)
        
        # Print at the beginning
        # print("[split_data] BEGIN: start_indices:", self.start_indices[:5], "...", self.start_indices[-5:] if len(self.start_indices) >= 5 else self.start_indices)
        # print("[split_data] BEGIN: start_rows:", self.start_rows[:5], "...", self.start_rows[-5:] if len(self.start_rows) >= 5 else self.start_rows)
        
        tot = len(self.start_indices)
        num_train = int(tot * 0.7)
        num_val = int(tot * 0.1)
        # get indices of first and last startof the current split in start_indices 
        split_indices = np.array([0, num_train, num_train + num_val, tot])
        ind1, ind2 = split_indices[self.set_type], split_indices[self.set_type + 1]-1
        border1, border2 = self.start_rows[ind1], self.start_rows[ind2] + self.sample_len # split df = rows in [border1:border2)
        df_data = df.iloc[border1:border2] # the current split
        
        if self.use_datetime:
            self.data_stamp = self.data_stamp.iloc[border1:border2].values # loc uses index, iloc uses actual row numbers
        
        if self.scale:
            end_train = self.start_rows[num_train-1] + self.sample_len
            train_data = df.iloc[:end_train]
            self._scaler = StandardScaler()
            self._scaler.fit(train_data.values)
            self.data = self._scaler.transform(df_data.values)
        else:
            self.data = df_data.values

        self.start_indices = self.start_indices[ind1:ind2+1] # to include ind2
        self.start_rows = self.start_rows[ind1:ind2+1] # = df_data.index.get_indexer(self.start_indices)
        self.indices = self.indices[border1:border2]
        # Print at the end
        # print("[split_data] END: start_indices:", self.start_indices[:5], "...", self.start_indices[-5:] if len(self.start_indices) >= 5 else self.start_indices)
        # print("[split_data] END: start_rows:", self.start_rows[:5], "...", self.start_rows[-5:] if len(self.start_rows) >= 5 else self.start_rows)
    
        
    def __getitem__(self, ind): 
        # ind is a random integer in range(len(self.start_rows))
        i0 = self.start_indices[ind] #start index (index in the original df)
        # find row number of i0 in the splitted df
        s0 = np.searchsorted(self.indices, i0) # row number in split df = row number of i0 in indices
        s1 = s0 + self.seq_len
        r0 = s1 - self.label_len
        r1 = s1 + self.pred_len
        # Get input sequence
        seq_x = self.data[s0:s1]
        seq_y = self.data[r0:r1]
        # timestamps might not be consecutive due to missing rows, so cannot take range(seq_len)
        if self.use_datetime:
            x_mark = self.data_stamp[s0:s1]
            y_mark = self.data_stamp[r0:r1]
        else:
            x_mark = self.indices[s0:s1] 
            y_mark = self.indices[r0:r1] 
        return seq_x, seq_y, x_mark, y_mark
    
    def __len__(self):
        return len(self.start_rows)
    
    def inverse_transform(self, data, index): # index is not used but just to match the input of CombinedDataset
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
        perc_missing (float): Maximum percentage of missing values allowed in a sequence
        chunksize (int): Maximum size of consecutive missing chunks allowed
        colnames (list): List of column names to use
        use_datetime (bool): Whether to use datetime features
    """
    def __init__(self, rootpath, filenames, flag='train', size=None, scale=True, 
                 perc_missing=5, chunksize=3, colnames=None, use_datetime=False):
        self.rootpath = rootpath
        self.filenames = filenames
        self.flag = flag
        self.size = size
        self.scale = scale
        self.perc_missing = perc_missing
        self.chunksize = chunksize
        self.colnames = colnames if colnames is not None else ['open', 'high', 'low', 'close', 'volume', 'log_return']
        self.use_datetime = use_datetime
        
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
                scale=scale,
                perc_missing=perc_missing,
                chunksize=chunksize,
                colnames=colnames,
                use_datetime=use_datetime
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
    
    def get_dataset_idx(self, index): # know which dataset the sample is from
        """Helper method to get the dataset index and local index for a given global index"""
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1 and index >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
        local_index = index - self.cumulative_lengths[dataset_idx]
        return dataset_idx, local_index
    
    def inverse_transform(self, data, indices=None):
        """
        Inverse transform the data using the appropriate scaler for each dataset.
        
        Args:
            data: The data to inverse transform
            indices: Optional array of indices corresponding to the data points.
                    If provided, will use the correct scaler for each data point.
                    If not provided, assumes all data is from the first dataset.
        """
        if indices is None:
            # If no indices provided, use first dataset's scaler (backward compatibility)
            return self.datasets[0].inverse_transform(data)
        
        # Create output array of same shape as input
        result = np.zeros_like(data)
        
        # Group data points by their dataset
        for dataset_idx in range(len(self.datasets)):
            # Find which data points belong to this dataset
            mask = np.zeros(len(indices), dtype=bool)
            for i, idx in enumerate(indices):
                d_idx, _ = self.get_dataset_idx(idx)
                mask[i] = (d_idx == dataset_idx)
            
            if np.any(mask):
                # Transform the data points for this dataset
                result[mask] = self.datasets[dataset_idx].inverse_transform(data[mask])
        
        return result
    
# def main():
#     # Test parameters
#     rootpath = "data"  # Change this to your data directory
#     filename = "XBTUSD_60.parquet"  # Change this to your data file
#     timestep = 3600  # 1 hour in seconds
    
#     # Test different sequence lengths
#     test_sizes = [
#         (24, 12, 4),    # 24 hours input, 12 hours overlap, 6 hours prediction
#         (168, 24, 12),   # 48 hours input, 24 hours overlap, 12 hours prediction
#         (720, 48, 24)    # 96 hours input, 48 hours overlap, 24 hours prediction
#     ]
    
#     print("\nTesting OHLCV_Dataset functionality:")
#     print("=====================================")
    
#     for size in test_sizes:
#         print(f"\nTesting with size parameters: {size}")
#         print("-" * 40)
        
#         # Initialize dataset
#         dataset = OHLCV_Dataset(
#             rootpath=rootpath,
#             filename=filename,
#             timestep=timestep,
#             flag='train',
#             size=size,
#             scale=True,
#             perc_missing=5,
#             chunksize=3
#         )
        
#         # Print dataset information
#         print(f"Total number of valid sequences: {len(dataset)}")
#         print(f"First 5 start indices: {dataset.start_indices[:5]}")
#         print(f"First 5 start rows: {dataset.start_rows[:5]}")
        
#         # Test __getitem__ for first few indices
#         print("\nTesting __getitem__ for first 3 indices:")
#         for i in np.random.choice(len(dataset), 3, replace=False):
#             seq_x, seq_y, x_mark, y_mark = dataset[i]
#             print(f"\nIndex {i}:")
#             print(f"Input sequence shape: {seq_x.shape}")
#             print(f"Target sequence shape: {seq_y.shape}")
#             print(f"X_MARK: {x_mark[:5]}...")  # Show first 5 timestamps
#             print(f"Y_MARK: {y_mark[:5]}...")  # Show first 5 timestamps
            
#             # Print some sample values
#             print("\nSample values:")
#             print("Input sequence first row:", seq_x[0])
#             print("Target sequence first row:", seq_y[0])

# if __name__ == "__main__":
#     main()
    