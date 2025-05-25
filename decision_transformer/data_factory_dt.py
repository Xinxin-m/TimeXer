from decision_transformer.data_loader_dt import OHLCV_Dataset, CombinedDataset
from torch.utils.data import DataLoader
import numpy as np


class IndexedDataLoader(DataLoader):
    """
    A DataLoader that yields (batch_indices, batch_data) for each batch.
    """
    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[i] for i in batch_indices]
            yield batch_indices, self.collate_fn(batch)

def get_dataloader(args, flag):
    """
    Data provider for Decision Transformer.
    Returns: dataset, dataloader (yields (indices, batch_x, batch_y, batch_x_mark, batch_y_mark))
    """
    shuffle_flag = False if (flag == 'test') else True
    batch_size = args.batch_size

    # Create dataset
    if hasattr(args, 'filenames') and args.filenames:
        # Multi-file mode
        dataset = CombinedDataset(
            rootpath=args.rootpath,
            filenames=args.filenames,
            timestep=args.timestep,
            use_datetime=args.use_datetime,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scale=True
        )
    else:
        # Single file mode
        # def __init__(self, rootpath, filename, timestep=3600, flag='train', size=None, scale=True, 
        # perc_missing=5, chunksize=3, colnames=None, use_datetime=True)
        dataset = OHLCV_Dataset(
            rootpath=args.rootpath,
            filename=args.filename,
            timestep=args.timestep,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            scale=True,
            perc_missing=args.perc_missing,
            chunksize=args.chunksize,
            use_datetime=args.use_datetime
        )

    # Create dataloader
    dataloader = IndexedDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False
    )

    return dataset, dataloader 