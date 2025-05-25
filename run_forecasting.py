import argparse
import os
import torch
import random
import numpy as np
from forecasting import OHLCV_Forecast

if __name__ == '__main__':
    # Set random seeds for reproducibility
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='OHLCV Forecasting')

    # Data parameters
    parser.add_argument('--rootpath', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--filename', type=str, default='XBTUSD_60.parquet', help='single data file name')
    parser.add_argument('--filenames', type=str, nargs='+', default=None, help='list of data file names')
    parser.add_argument('--timestep', type=int, default=3600, help='timestep for the data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--perc_missing', type=float, default=5, help='percentage of missing values allowed in a sequence')
    parser.add_argument('--chunksize', type=int, default=3, help='maximum size of consecutive missing chunks allowed')
    parser.add_argument('--use_datetime', type=bool, default=True, help='use datetime features')
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') # Original transformer: d_ff=2048, d_model = 512
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=24, help='patch length')
    parser.add_argument('--normalize', type=bool, default=False, help='whether to normalize the current data sample in model.forward()')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment type') # type1: lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1)

    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--is_training', type=int, default=1, help='status: 1 for training, 0 for testing')

    # GPU parameters
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # Validate data parameters
    if args.filename is None and args.filenames is None:
        raise ValueError("Either filename or filenames must be provided")

    # Create experiment setting name
    setting = 'multi' if args.filenames else 'single'
    # setting = '{}_sl{}_ll{}_pl{}_dm{}_h{}_enc{}_dec{}_dff{}_{}'.format(
    #     'multi' if args.filenames else 'single',
    #     args.seq_len,
    #     args.label_len,
    #     args.pred_len,
    #     args.d_model,
    #     args.n_heads,
    #     args.e_layers,
    #     args.d_layers,
    #     args.d_ff,
    #     args.des
    # )

    print('Args in experiment:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # Initialize forecaster
    forecaster = OHLCV_Forecast(args)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        forecaster.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        forecaster.test(setting)
        torch.cuda.empty_cache()
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        forecaster.test(setting, test=1)
        torch.cuda.empty_cache() 