import torch
import torch.nn as nn
from models.TimeXer_time import Model
from forecasting import OHLCV_Forecast
import argparse

def test_data_split(forecaster, split_name):
    print(f"\n=== Testing {split_name} data ===")
    data, loader = forecaster._get_data(flag=split_name)
    
    print(f"Dataset length: {len(data)}")
    
    # Test a single batch
    for batch_indices, batch_data in loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
        
        print(f"\nBatch shapes:")
        print(f"batch_x shape: {batch_x.shape}")
        print(f"batch_y shape: {batch_y.shape}")
        print(f"batch_x_mark shape: {batch_x_mark.shape}")
        print(f"batch_y_mark shape: {batch_y_mark.shape}")
        
        # Check if tensors are empty
        print("\nEmpty tensor check:")
        print(f"batch_x empty: {batch_x.numel() == 0}")
        print(f"batch_y empty: {batch_y.numel() == 0}")
        print(f"batch_x_mark empty: {batch_x_mark.numel() == 0}")
        print(f"batch_y_mark empty: {batch_y_mark.numel() == 0}")
   
        break  # Only test first batch

def test_validation():
    # Create a minimal args object with required parameters
    args = argparse.Namespace(
        rootpath='./data/',
        filename='XBTUSD_60.parquet',
        timestep=3600,
        batch_size=32,
        num_workers=0,
        seq_len=168,
        label_len=48,
        pred_len=24,
        inverse=False,
        perc_missing=5,
        chunksize=3,
        use_datetime=True,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.1,
        activation='gelu',
        patch_len=24,
        normalize=True,
        lradj='type1',
        train_epochs=10,
        patience=3,
        learning_rate=0.0001,
        checkpoints='./checkpoints/',
        is_training=1,
        use_gpu=False,
        gpu=0
    )

    # Initialize forecaster
    forecaster = OHLCV_Forecast(args)
    
    # Test all splits
    test_data_split(forecaster, 'train')
    test_data_split(forecaster, 'val')
    test_data_split(forecaster, 'test')

if __name__ == "__main__":
    test_validation() 