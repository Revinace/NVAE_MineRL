# The Official PyTorch Implementation of "NVAE: A Deep Hierarchical Variational Autoencoder" [(NeurIPS 2020 Spotlight Paper)](https://arxiv.org/abs/2007.03898)

<div align="center">
  <a href="http://latentspace.cc/arash_vahdat/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://jankautz.com/" target="_blank">Jan&nbsp;Kautz</a> 
</div>

##Disclaimer
Currently I only have evidence, that the creation of the dataset and the training is working on Linux because windows have some disadvantages.
When you only want to work with the pre_trained files windows should work when the `args.distribution` is `False` and `num_workers` in `datasets` is 0,
which is automatically the case if `args.OS` is `windows`.

---

## Requirements
NVAE is built in Python 3.7 using PyTorch 1.6.0. Use the following command to install the requirements:
```
pip install -r requirements.txt
``` 
If you have problems to install `MineRL` try the following
```
sudo add-apt-repository -y ppa:openjdk-r/ppa
sudo apt-get purge openjdk-*
sudo apt-get install openjdk-8-jdk
sudo apt-get install xvfb xserver-xephyr vnc4server python-opengl ffmpeg
pip3 install --upgrade minerl
```

---

## Structure
The folder for the [MineRL](https://minerl.readthedocs.io/en/latest/) enviourments is `data/`

The LMDB Datasets are stored in `datasets/minecraft_lmdb` and the pretrained model in `results/eval-minecraft/checkpoint.pt`

The model and pretrained weights for the [Critic Guided Segmentation](https://arxiv.org/abs/2107.09540) are saved in the folder `critic_guided_segmentation/`

The argument `--OS` was introduced because windows seems to have problems with multiprocessing and the pytorch distribution library.
Here the main difference is that `args.distribution` is `False` and the `num_workers` in `datasets.py` are 0.

The notebook `NVAE_Dataset_Train_Sample.ipynb` or `dataset_train_script.py` can be used to create a dataset, train it and sample a few images.

---

<details><summary>Create Dataset and train the NVAE</summary>

## Create Minecraft Dataset
Run the following commands to generate the Minecraft images and store them in an LMDB dataset:

```shell script
python scripts/create_custom_lmdb_from_minerl.py --lmdb_path datasets/minecraft_lmdb --train_size 15000 --test_size 5000
```
The LMDB datasets are created at `datasets/minecraft`.
**Important info for Windows user**: the dataset is only on Linux able to shrink after processing, so for windows the `mapsize` in this case 10GB will be used.


## Running the training of NVAE for Minecraft
Before the training can start the `train_size` and `test_size` need to be adjusted in `lmdb_datasets.py`
Currently only the default parameters from the NVAE where used to train the network.
Maybe it's helpfull to use one of the [settings](https://github.com/NVlabs/NVAE#running-the-main-nvae-training-and-evaluation-scripts) used for the face datasets.

```shell script
python train.py --data datasets/minecraft_lmdb --root results --save minecraft --dataset minecraft
```

**If for any reason your training is stopped, use the exact same commend with the addition of `--cont_training`
to continue training from the last saved checkpoint. If you observe NaN, continuing the training using this flag
usually will not fix the NaN issue.**

</details>

<details><summary>Optimize Training by Nvidia</summary>

## How to construct smaller NVAE models
In the commands above, we are constructing big NVAE models that require several days of training
in most cases. If you'd like to construct smaller NVAEs, you can use these tricks:

* Reduce the network width: `--num_channels_enc` and `--num_channels_dec` are controlling the number
of initial channels in the bottom-up and top-down networks respectively. Recall that we halve the
number of channels with every spatial downsampling layer in the bottom-up network, and we double the number of
channels with every upsampling layer in the top-down network. By reducing
`--num_channels_enc` and `--num_channels_dec`, you can reduce the overall width of the networks.

* Reduce the number of residual cells in the hierarchy: `--num_cell_per_cond_enc` and 
`--num_cell_per_cond_dec` control the number of residual cells used between every latent variable
group in the bottom-up and top-down networks respectively. In most of our experiments, we are using
two cells per group for both networks. You can reduce the number of residual cells to one to make the model
smaller.

* Reduce the number of epochs: You can reduce the training time by reducing `--epochs`.

* Reduce the number of groups: You can make NVAE smaller by using a smaller number of latent variable groups. 
We use two schemes for setting the number of groups:
    1. An equal number of groups: This is set by `--num_groups_per_scale` which indicates the number of groups 
    in each scale of latent variables. Reduce this number to have a small NVAE.
    
    2. An adaptive number of groups: This is enabled by `--ada_groups`. In this case, the highest
    resolution of latent variables will have `--num_groups_per_scale` groups and 
    the smaller scales will get half the number of groups successively (see groups_per_scale in utils.py).
    We don't let the number of groups go below `--min_groups_per_scale`. You can reduce
    the total number of groups by reducing `--num_groups_per_scale` and `--min_groups_per_scale`
    when `--ada_groups` is enabled.

</details> 

<details><summary>License</summary>

## License
Please check the LICENSE file. NVAE may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

You should take into consideration that VAEs are trained to mimic the training data distribution, and, any 
bias introduced in data collection will make VAEs generate samples with a similar bias. Additional bias could be 
introduced during model design, training, or when VAEs are sampled using small temperatures. Bias correction in 
generative learning is an active area of research, and we recommend interested readers to check this area before 
building applications using NVAE.

</details>

<details><summary>Bibtex</summary>

## Bibtex:
Please cite our paper, if you happen to use this codebase:

```
@inproceedings{vahdat2020NVAE,
  title={{NVAE}: A Deep Hierarchical Variational Autoencoder},
  author={Vahdat, Arash and Kautz, Jan},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

</details>
