import os

dataset_path = "datasets"
checkpoint_dir = "results"
system = "linux"
os.system("python scripts/create_custom_lmdb_from_minerl.py --custom_lmdb_path {}/minecraft_lmdb".format(dataset_path))
os.system("python train.py --data {}/minecraft_lmdb --root {} --save minecraft --dataset minecraft --OS {} --batch_size 100".format(dataset_path, checkpoint_dir, system))
os.system("python sample_images.py --OS {} --batch_size 1".format(system))