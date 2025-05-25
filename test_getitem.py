import torch
from decision_transformer.data_loader_dt import OHLCV_Dataset
import argparse

'''
Test the getitem method in data_loader_dt (compare the values with the values in the original df)
'''
def test_getitem():
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
        scale=False,
        lradj='type1',
        train_epochs=10,
        patience=3,
        learning_rate=0.0001,
        checkpoints='./checkpoints/',
        is_training=1,
        use_gpu=False,
        gpu=0
    )

    # Test all splits
    for flag in ['train', 'val', 'test']:
        print(f"\n=== Testing {flag} data ===")
        
        # Initialize dataset
        dataset = OHLCV_Dataset(
            rootpath=args.rootpath,
            filename=args.filename,
            timestep=args.timestep,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scale=args.scale,
            perc_missing=args.perc_missing,
            chunksize=args.chunksize,
            use_datetime=args.use_datetime,
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test first and last indices
        for idx in [0, len(dataset)-1]:
            print(f"\nTesting index {idx}:")
            
            # Get the item
            seq_x, seq_y, x_mark, y_mark = dataset[idx]
            
            # Get the internal indices used
            i0 = dataset.start_indices[idx]
            s0 = dataset.indices.searchsorted(i0)
            s1 = s0 + dataset.seq_len
            r0 = s1 - dataset.label_len
            r1 = s1 + dataset.pred_len
            
            print(f"i0 (start index): {i0}")
            print(f"s0: {s0}")
            print(f"s1: {s1}")
            print(f"r0: {r0}")
            print(f"r1: {r1}")
            
            # Print first entry of indices and data_stamp
            print(f"\nFirst entry of indices: {dataset.indices[0]}")
            print(f"First entry of data_stamp: {dataset.data_stamp[0]}")
            print(f"First entry of data_x: {seq_x[0]}")

if __name__ == "__main__":
    test_getitem() 