import os

dataset_path = "datasets"
checkpoint_dir = "results"
#os.system("python scripts/create_custom_lmdb_from_minerl.py --custom_lmdb_path {}/minecraft_lmdb".format(dataset_path))
os.system("python train.py --data {}/minecraft_lmdb --root {} --save minecraft --dataset minecraft --OS win".format(dataset_path, checkpoint_dir))
#os.system("python sample_images.py")