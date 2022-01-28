# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
import lmdb
import os
import matplotlib
import minerl
from minerl.data import BufferedBatchIter
from tqdm import tqdm
import sys

from PIL import Image

MINECRAFT_ENVIRONMENT = "MineRLTreechop-v0"

# YOU ONLY NEED TO DO THIS ONCE!
os.makedirs("data", exist_ok=True)
data_root = os.getenv('MINERL_DATA_ROOT', 'data/')
if MINECRAFT_ENVIRONMENT not in os.listdir(data_root):
    print(f"Downloading {MINECRAFT_ENVIRONMENT}")
    minerl.data.download(directory=data_root, environment=MINECRAFT_ENVIRONMENT)


def main(custom_lmdb_path):

    # create target directory
    if not os.path.exists(custom_lmdb_path):
        os.makedirs(custom_lmdb_path, exist_ok=True)

    lmdb_path_train = os.path.join(custom_lmdb_path, 'train.lmdb')
    lmdb_path_validation = os.path.join(custom_lmdb_path, 'validation.lmdb')

    data = minerl.data.make(environment=MINECRAFT_ENVIRONMENT, data_dir=data_root)
    iterator = BufferedBatchIter(data)
    for idx, order in enumerate([args.train_order, args.test_order]):
        count = 0
        if idx==0:
            env = lmdb.open(lmdb_path_train, map_size=1e10)
            max_size = args.train_size
            output_string = "train"
        else:
            env = lmdb.open(lmdb_path_validation, map_size=1e10)
            max_size = args.test_size
            output_string = "test"
        if(order == "random"):
            with env.begin(write=True) as txn:
                # random reading of MineRL Enviourment
                for current_state, _, _, _, _ in tqdm(iterator.buffered_batch_iter(batch_size=1, num_epochs=1)):
                    if (count >= max_size):
                        break
                    x = current_state['pov'][0].astype(np.uint8)
                    im = Image.fromarray(x)
                    im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)
                    txn.put(str(count).encode(), im)
                    count += 1
                print('added {} items to the LMDB {} dataset.'.format(count, output_string))
            env.close()

        else:
            with env.begin(write=True) as txn:
                # sequential reading of MineRL Enviourment
                for current_state, _, _, _, _ in tqdm(data.batch_iter(batch_size=1, num_epochs=1, seq_len=1)):
                    if (count >= max_size):
                        break
                    x = current_state['pov'][0][0].astype(np.uint8)
                    im = Image.fromarray(x)
                    im = np.array(im.getdata(), dtype=np.uint8).reshape(im.size[1], im.size[0], 3)
                    txn.put(str(count).encode(), im)
                    count += 1
                print('added {} items to the LMDB {} dataset.'.format(count, output_string))
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom Training Set')
    parser.add_argument('--custom_lmdb_path', type=str, default='datasets/minecraft_lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--train_size', type=int, default=200)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('--train_order', type=str, default="random", choices={"random", "sequential"})
    parser.add_argument('--test_order', type=str, default="random", choices={"random", "sequential"})
    args = parser.parse_args()

    main(args.custom_lmdb_path)

