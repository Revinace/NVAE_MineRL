# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse

import torch
import numpy as np
import os
from tqdm import tqdm
import pickle

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import GradScaler
import torchvision

from model import AutoEncoder
from thirdparty.adamax import Adamax
import utils
import datasets
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from PIL import Image, ImageDraw

from critic_guided_segmentation.critique import NewCritic


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def main(args):
    kmean_latent_cluster_size = args.kmean_latent_cluster_size
    image_number_clustering = args.image_number_clustering
    use_latent_filter = args.use_latent_filter
    load_kmeans_latent = args.load_kmeans_latent

    # ensures that weight initializations are all the same

    save_path = "critic_guided_segmentation/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=100000-shift=12-chfak=1-dropout=0.3.pt"
    critic = NewCritic(bottleneck=32, chfak=1, dropout=0.3)
    critic.load_state_dict(torch.load(save_path))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

   # Get data loaders.
    train_queue , valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, writer, arch_instance)

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    logging.info('loading the model.')
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    init_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    cnn_optimizer.load_state_dict(checkpoint['optimizer'])
    grad_scalar.load_state_dict(checkpoint['grad_scalar'])
    cnn_scheduler.load_state_dict(checkpoint['scheduler'])
    global_step = checkpoint['global_step']
    model.eval()

    if args.distributed:
        if (not (args.OS == "windows") and not (args.OS == "win")):
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

    os.makedirs("images", exist_ok=True)

    image_list = []
    latent_list = []
    for step, x in enumerate(tqdm(train_queue)):
        x = x[0] if len(x) > 1 else x
        image = x
        x = x.cuda()
        x = utils.pre_process(x, args.num_x_bits)
        logits, log_q, log_p, kl_all, kl_diag, latent, pre_layers = model(x)
        if(step*args.batch_size_trainset < image_number_clustering):
            image_list.extend(list(image.detach().cpu().numpy()))
            latent_list.append(latent.detach().cpu().numpy())
        else:
            break
    arr_latent = np.stack(latent_list, axis=0)
    arr_latent = arr_latent.reshape(arr_latent.shape[0]*arr_latent.shape[1], -1)

    # KMeans of feature_layers
    if(use_latent_filter):
        print("KMean_Latent")
        path = "clusters/cluster_std/Cluster_Histogram_{}".format(kmean_latent_cluster_size)
        os.makedirs(path, exist_ok=True)
        if(load_kmeans_latent):
            with open(path + "/kmeans_latent_{}.pkl".format(kmean_latent_cluster_size), "rb") as f:
                kmeans_latent = pickle.load(f)
            kml = kmeans_latent.predict(arr_latent)
        else:
            kmeans_latent = KMeans(n_clusters=kmean_latent_cluster_size, random_state=42).fit(arr_latent)
            with open(path + "/kmeans_latent_{}.pkl".format(kmean_latent_cluster_size), "wb") as f:
                pickle.dump(kmeans_latent, f)
            kml = kmeans_latent.labels_
        print(np.histogram(kml, bins=kmean_latent_cluster_size)[0])
        critic_cluster = {}
        images_in_cluster = {}

        for idx in range(image_number_clustering):
            image = image_list[idx]
            shape = image.shape
            XP = torch.tensor(image.reshape(1, shape[0], shape[1], shape[2])).float()
            pred = critic(XP).squeeze()
            if(kml[idx] not in critic_cluster.keys()):
                critic_cluster[kml[idx]] = []
                images_in_cluster[kml[idx]] = []
            critic_cluster[kml[idx]].append(pred.detach().numpy())
            images_in_cluster[kml[idx]].append(image)

        fig, ax = plt.subplots()
        for key,value in critic_cluster.items():
            images = np.stack(images_in_cluster[key])
            critics = np.stack(value)
            images_a_idx = np.where((critics > 0.5))[0]
            images_a = images[images_a_idx]
            critics_a = critics[images_a_idx]
            if(images_a.shape[0] == 0):
                images_a = np.zeros((1,3,64,64))
                critics_a = np.zeros((1))
            images_idx_a = np.random.randint(images_a.shape[0], size=100)
            grid = torchvision.utils.make_grid(torch.tensor(images_a[images_idx_a]), 10)
            grid = grid.detach().cpu().numpy().transpose(1, 2, 0) * 255
            img_a = Image.fromarray(grid.astype("uint8"))
            draw = ImageDraw.Draw(img_a)
            # print(draw.textsize(str(1)))
            for i in range(10):
                for j in range(10):
                    x, y = int(2 + (j * (img_a.width / 10))), int(2 + (i * (img_a.height / 10)))
                    draw.text((x, y), str(round(critics_a[images_idx_a][i*10+j], 3)), fill=(255, 255, 255), align="left")

            hist = np.histogram(critics, bins=20, range=(0.0, 1.0))
            ax.clear()
            ax.bar(np.arange(0,1,0.05), hist[0], width=0.05, align='edge')
            ax.set_box_aspect(1)
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            img_hist = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_hist = img_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_hist = Image.fromarray(img_hist)
            img_hist = img_hist.resize((int(img_hist.width*(img_a.height/img_hist.height)), img_a.height))

            images_b_idx = np.where((critics <= 0.5))[0]
            images_b = images[images_b_idx]
            critics_b = critics[images_b_idx]
            if(images_b.shape[0] == 0):
                images_b = np.zeros((1,3,64,64))
                critics_b = np.zeros((1))
            images_idx_b = np.random.randint(images_b.shape[0], size=100)
            grid = torchvision.utils.make_grid(torch.tensor(images_b[images_idx_b]), 10)
            grid = grid.detach().cpu().numpy().transpose(1, 2, 0) * 255
            img_b = Image.fromarray(grid.astype("uint8"))
            draw = ImageDraw.Draw(img_b)
            # print(draw.textsize(str(1)))
            for i in range(10):
                for j in range(10):
                    x, y = int(2 + (j * (img_b.width / 10))), int(2 + (i * (img_b.height / 10)))
                    draw.text((x, y), str(round(critics_b[images_idx_b][i*10+j], 3)), fill=(255, 255, 255), align="left")

            img_out = get_concat_h(img_a, img_hist)
            img_out = get_concat_h(img_out, img_b)

            img_out.save(path + "/{}.png".format(key))


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # custom
    parser.add_argument('--return_feature_layers', default=True)
    parser.add_argument('--OS', choices=['windows', 'win', 'linux', 'ubuntu'], default='windows')
    parser.add_argument('--kmean_latent_cluster_size', default=64, type=int)
    parser.add_argument('--kmean_pixel_cluster_size', default=128, type=int)
    parser.add_argument('--image_number_clustering', default=15000, type=int)
    parser.add_argument('--render_start_idx', default=0, type=int)
    parser.add_argument('--render_stop_idx', default=500, type=int)
    parser.add_argument('--use_latent_filter', action='store_true', default=True)
    parser.add_argument('--load_kmeans_latent', action='store_true', default=False)
    parser.add_argument('--batch_size_trainset', type=int, default=8)
    parser.add_argument('--batch_size_testset', type=int, default=1)
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64', 'minecraft'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if (args.OS == "windows" or args.OS == "win"):
        # for debugging
        print('starting in debug mode')
        args.distributed = False
        main(args)
    else:
        if size > 1:
            args.distributed = True
            processes = []
            for rank in range(size):
                args.local_rank = rank
                global_rank = rank + args.node_rank * args.num_process_per_node
                global_size = args.num_proc_node * args.num_process_per_node
                args.global_rank = global_rank
                print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
                p = Process(target=init_processes, args=(global_rank, global_size, main, args))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            # for debugging
            print('starting in debug mode')
            args.distributed = True
            init_processes(0, size, main, args)


