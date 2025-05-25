import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def find_nan_rows(filepath, column='log_return', n_rows=0):
    """
    Find and print rows where the specified column has NaN values.
    
    Args:
        filepath (str): Path to the data file (csv or parquet)
        column (str): Column name to check for NaN values
        n_rows (int): Number of rows to print before and after the NaN value
    """
    print(f"\nFinding NaN values in column '{column}' for file: {filepath}")
    
    # Read the file
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Find rows with NaN values
    nan_mask = df[column].isna()
    nan_indices = df.index[nan_mask].tolist()
    
    if len(nan_indices) == 0:
        print(f"No NaN values found in column '{column}'")
        return
    
    print(f"\nFound {len(nan_indices)} NaN values in column '{column}'")
    
    # For each NaN value, print surrounding rows
    for idx in nan_indices:
        start_idx = max(0, idx - n_rows)
        end_idx = min(len(df), idx + n_rows + 1)
        
        print(f"\nNaN value at index {idx}:")
        print("Surrounding rows:")
        print(df.iloc[start_idx:end_idx])
        print("-" * 80)

def check_data_quality(filepath, seq_len=720, pred_len=24, scale=True):
    """
    Check data quality of a file by examining for NaN values and basic statistics.
    Uses the same data splitting logic as OHLCV_Dataset.
    
    Args:
        filepath (str): Path to the data file (csv or parquet)
        seq_len (int): Sequence length for splitting data
        pred_len (int): Prediction length
        scale (bool): Whether to check scaled data as well
        
    Returns:
        dict: Dictionary containing data quality metrics
    """
    print(f"\nChecking data quality for file: {filepath}")
    
    # Read the file
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
        
    # Check for NaN values
    nan_counts = df.isna().sum()
    total_rows = len(df)
    
    print("\nNaN counts per column:")
    for col in df.columns:
        nan_count = nan_counts[col]
        nan_percentage = (nan_count / total_rows) * 100
        print(f"{col}: {nan_count} NaN values ({nan_percentage:.2f}%)")
        
    # Basic statistics for numeric columns
    print("\nBasic statistics for numeric columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    print(stats)
    
    # Check for infinite values
    inf_counts = np.isinf(df[numeric_cols]).sum()
    print("\nInfinite values per column:")
    for col in numeric_cols:
        inf_count = inf_counts[col]
        inf_percentage = (inf_count / total_rows) * 100
        print(f"{col}: {inf_count} infinite values ({inf_percentage:.2f}%)")
        
    # Check for zeros in volume
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        print(f"\nZero volume entries: {zero_volume} ({zero_volume/total_rows*100:.2f}%)")
        
    # Check for negative values in OHLC
    ohlc_cols = ['open', 'high', 'low', 'close']
    for col in ohlc_cols:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            print(f"Negative/zero {col} values: {neg_count} ({neg_count/total_rows*100:.2f}%)")
            
    # Check for high-low consistency
    if all(col in df.columns for col in ['high', 'low']):
        invalid_hl = (df['high'] < df['low']).sum()
        print(f"\nInvalid high-low pairs: {invalid_hl} ({invalid_hl/total_rows*100:.2f}%)")
        
    # Check for OHLC consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = ((df['high'] < df['open']) | 
                      (df['high'] < df['close']) | 
                      (df['low'] > df['open']) | 
                      (df['low'] > df['close'])).sum()
        print(f"Invalid OHLC combinations: {invalid_ohlc} ({invalid_ohlc/total_rows*100:.2f}%)")
    
    # Split data into train/val/test using the same logic as OHLCV_Dataset
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_val = len(df) - num_train - num_test
    
    border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
    border2s = [num_train, num_train + num_val, len(df)]
    
    # Check each split
    splits = ['train', 'val', 'test']
    for i, (border1, border2) in enumerate(zip(border1s, border2s)):
        print(f"\nChecking {splits[i]} split (indices {border1}:{border2}):")
        split_data = df.iloc[border1:border2]
        
        # Check for NaN values in split
        split_nan = split_data.isna().sum()
        print(f"\nNaN counts in {splits[i]} split:")
        for col in split_data.columns:
            nan_count = split_nan[col]
            nan_percentage = (nan_count / len(split_data)) * 100
            print(f"{col}: {nan_count} NaN values ({nan_percentage:.2f}%)")
    
    # Check scaled data if requested
    if scale:
        print("\nChecking scaled data:")
        df_no_time = df.drop('timestamp', axis=1)
        
        # Scale using training data statistics
        train_data = df_no_time.iloc[border1s[0]:border2s[0]]
        scaler = StandardScaler()
        scaler.fit(train_data.values)
        
        # Check each split after scaling
        for i, (border1, border2) in enumerate(zip(border1s, border2s)):
            split_data = df_no_time.iloc[border1:border2]
            scaled_data = scaler.transform(split_data.values)
            
            # Check for NaN values after scaling
            nan_mask = np.isnan(scaled_data)
            if nan_mask.any():
                print(f"\nWARNING: NaN values detected in scaled {splits[i]} data!")
                for col in range(scaled_data.shape[1]):
                    nan_count = nan_mask[:, col].sum()
                    if nan_count > 0:
                        print(f"Column {col}: {nan_count} NaN values")
            
            # Check for infinite values after scaling
            inf_mask = np.isinf(scaled_data)
            if inf_mask.any():
                print(f"\nWARNING: Infinite values detected in scaled {splits[i]} data!")
                for col in range(scaled_data.shape[1]):
                    inf_count = inf_mask[:, col].sum()
                    if inf_count > 0:
                        print(f"Column {col}: {inf_count} infinite values")
            
            # Print basic statistics of scaled data
            print(f"\nBasic statistics of scaled {splits[i]} data:")
            scaled_df = pd.DataFrame(scaled_data, columns=df_no_time.columns)
            print(scaled_df.describe())
    
    return {
        'nan_counts': nan_counts.to_dict(),
        'stats': stats.to_dict(),
        'inf_counts': inf_counts.to_dict()
    }

def main():
    """
    Example usage of the data quality check function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Check data quality of OHLCV data file')
    parser.add_argument('--filepath', type=str, default='data/XBTUSD_60.parquet', help='Path to the data file (csv or parquet)')
    parser.add_argument('--seq_len', type=int, default=360, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    parser.add_argument('--scale', action='store_true', help='Check scaled data')
    parser.add_argument('--find_nan', default=True, help='Find and print rows with NaN values')
    parser.add_argument('--column', type=str, default='log_return', help='Column to check for NaN values')
    parser.add_argument('--n_rows', type=int, default=1, help='Number of rows to print before and after NaN values')
    
    args = parser.parse_args()
    
    if args.find_nan:
        find_nan_rows(
            filepath=args.filepath,
            column=args.column,
            n_rows=args.n_rows
        )
    else:
        check_data_quality(
            filepath=args.filepath,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            scale=args.scale
        )

if __name__ == "__main__":
    main() 