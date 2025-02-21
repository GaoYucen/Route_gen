# config.py

model_name = 'origin'

# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description="Pytorch implementation of Road-SPDE")

# Model
parser.add_argument('--num_samples', default=1000, type=int, help='Number of samples')
parser.add_argument('--test_sample_start_index', default=800, type=int, help='start of test samples')
parser.add_argument('--test_sample_end_index', default=900, type=int, help='end of test samples')
parser.add_argument('--nof_epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--hidden_dimen', default=512, type=int, help='Dimension of embedding')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--nof_samples', default=1000000, type=int, help='Number of samples')
parser.add_argument('--input_dimension', default=2, type=int, help='input dimension')
parser.add_argument('--embedding_dimen', default=128, type=int, help='embedding dimension')

parser.add_argument('--threshold', default=500, type=int, help='reach threshold')

# Data
parser.add_argument('--num', default=3, type=int, help='Data number')
parser.add_argument('--reverse_num', default=1000, type=int, help='Reverse Data number')
parser.add_argument('--dis_node_num', default=1000, type=int, help='Distance Data number')
parser.add_argument('--output_dimen', default=256, type=int, help='Output dimension')
parser.add_argument('--tildeL1_ratio', default=0.99, type=int, help='tildeL1 ratio')
parser.add_argument('--g_training_epochs', default=10, type=int, help='epochs of global training')

# Train
parser.add_argument('--num_epochs', default=100, type=int, help='max number epoch')
parser.add_argument('--patience', default=10, type=int, help='patience epochs')
parser.add_argument('--learning_rate', default=5e-3, type=int, help='learning rate')
parser.add_argument('--weight_decay', default=1e-3, type=int, help='weight_decay')

# NO Direct
parser.add_argument('--nodirect', default=False, type=bool, help='Whether to use no direct graph')
parser.add_argument('--sim', default=False, type=bool, help='Sim or not')
parser.add_argument('--city', default='chengdu_data', type=str, help='city name')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
