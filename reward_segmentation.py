# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import copy

import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import time

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
import cv2
from skimage import morphology

from critic_guided_segmentation.critique import NewCritic
from critic_guided_segmentation.UNet_Decoder import UnetDecoder
from FaissKNN import FaissKNeighbors


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
    image_number_clustering = 15000 #args.image_number_clustering
    render_start_idx = args.render_start_idx
    render_stop_idx = args.render_stop_idx
    use_latent_filter = args.use_latent_filter
    load_kmeans_latent = args.load_kmeans_latent
    set_size = 90
    ratio_threshold_list = [0.1,0.2,0.3]
    ratio_threshold = 0.4
    k_range = 1000
    layer_idx = 4
    load_images = True
    output_video = False

    # ensures that weight initializations are all the same

    critic_path = "critic_guided_segmentation/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=100000-shift=12-chfak=1-dropout=0.3.pt"
    critic = NewCritic(bottleneck=32, chfak=1, dropout=0.3)
    critic.load_state_dict(torch.load(critic_path))

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

    if(not load_images):
        image_list = []
        latent_list = []
        for step, x in enumerate(tqdm(train_queue)):
            x = x[0] if len(x) > 1 else x
            image = x
            x = x.cuda()
            x = utils.pre_process(x, args.num_x_bits)
            _, _, _, _, _, latent, pre_layers = model(x)
            #for layer in pre_layers:
            #    print(layer.shape)
            #exit()
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
            path = "clusters/cluster_mean_std/Cluster_Histogram_{}".format(kmean_latent_cluster_size)
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

            max_critic_latent = 11

            #max_critic_latent = np.argmax(counter[:,0])
            print(max_critic_latent)
            print(np.histogram(critic_cluster[max_critic_latent], bins=10)[0])

        list_A = []
        list_B = []
        list_image_A = []
        list_image_B = []
        counter_A = 0
        counter_B = 0
        print("seperate pixels by critic value into list_A and list_B")
        for idx in range(image_number_clustering):
            if (counter_A + counter_B == set_size * 2):
                break
            if (not use_latent_filter or kml[idx] == max_critic_latent):
                x = torch.tensor(image_list[idx])
                x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
                image = x
                x = x.cuda()
                x = utils.pre_process(x, args.num_x_bits)
                _, _, _, _, _, latent, pre_layers = model(x)
                XP = torch.tensor(image).float()
                pred = critic(XP).squeeze()
                layer = pre_layers[layer_idx][0].detach().cpu().numpy()
                # scale = int(64 / layer.shape[1])
                # layer = np.kron(layer, np.ones((scale,scale)))
                # combined_column = column np.concatenate((column, pl0, pl1))
                image_column_list = list(layer.reshape(layer.shape[0], -1).transpose(1, 0))
                if (pred > 0.75 and counter_A < set_size):
                    list_A = list_A + image_column_list
                    list_image_A.append(image[0].detach().cpu().numpy())
                    counter_A += 1
                if (pred < 0.75 and counter_B < set_size):
                    list_B = list_B + image_column_list
                    list_image_B.append(image[0].detach().cpu().numpy())
                    counter_B += 1
        print("Counter A : ", counter_A, "\tCounter B : ", counter_B)

        layer_res = pre_layers[layer_idx].shape[2]
        layer_dim = pre_layers[layer_idx].shape[1]

        arr_A = np.stack(list_A, axis=0).astype("float32")
        arr_B = np.stack(list_B, axis=0).astype("float32")
        arr_C = np.concatenate((arr_A, arr_B), axis=0)

        """xc = arr_C - arr_C.mean(0)
        cov = np.dot(xc.T, xc) / xc.shape[0]
        L = np.linalg.cholesky(cov)
        mahalanobis_transform = np.linalg.inv(L)
        y = np.dot(x, mahalanobis_transform.T)
        yc = y - y.mean(0)
        ycov = np.dot(yc.T, yc) / yc.shape[0]"""

        arr_image_A = np.stack(list_image_A, axis=0).astype("float32")
        arr_image_B = np.stack(list_image_B, axis=0).astype("float32")

        data = [arr_A, arr_B, arr_image_A, arr_image_B]
        np.savez("number{}_layer{}_images.npz".format(set_size, layer_idx), *data)
    else:
        try:
            container = np.load("number{}_layer{}_images.npz".format(set_size, layer_idx))
            [arr_A, arr_B, arr_image_A, arr_image_B] = [container[key] for key in container]
            arr_C = np.concatenate((arr_A, arr_B), axis=0)
            layer_dim = arr_A.shape[1]
        except:
            print("Missing .npz file. Set load_images to False.")
            return -1

    grid_A = torchvision.utils.make_grid(torch.tensor(arr_image_A))
    plt.imshow(grid_A.detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()
    grid_B = torchvision.utils.make_grid(torch.tensor(arr_image_B))
    plt.imshow(grid_B.detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()

    s0 = time.time()
    print("compute distance to kth nearest neighbour")
    faissknn = FaissKNeighbors()
    faissknn.fit(arr_C)

    #use to create list of regions that fullfill the ratio condition
    """distances, indices = faissknn.predict(arr_C, k=k_range)
    radius = distances[:, -1]
    indices_b = np.sum(np.where(indices >= arr_A.shape[0], 1, 0), axis=1)
    ratio = np.divide(indices_b, k_range)
    special_indices = np.where(ratio < ratio_threshold, np.arange(arr_C.shape[0]), -1)
    special_indices = np.delete(special_indices, np.where(special_indices == -1))
    radius = radius[special_indices]
    faissknn_special = FaissKNeighbors()
    faissknn_special.fit(arr_C[special_indices])
    lower_bound = arr_C[special_indices] - np.repeat(radius.reshape(-1,1), layer_dim, axis=1)
    upper_bound = arr_C[special_indices] + np.repeat(radius.reshape(-1,1), layer_dim, axis=1)
    clusters = list(zip(lower_bound, upper_bound))"""

    s1 = time.time()

    print(s1 - s0)

    print("Render Images")
    path = "image_density_kmean{}_imagenumber{}_new/k_{}/layer_{}".format(kmean_latent_cluster_size, image_number_clustering, k_range, layer_idx)
    os.makedirs(path, exist_ok=True)

    for idx, image in enumerate(tqdm(np.concatenate((arr_image_A,arr_image_B), axis=0))):
        x = torch.tensor(image)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        #image = x
        x = x.cuda()
        x = utils.pre_process(x, args.num_x_bits)
        _, _, _, _, _, latent, pre_layers = model(x)
        layer = pre_layers[layer_idx][0].detach().cpu().numpy()
        scale = int(64 / layer.shape[1])
        layer = np.kron(layer, np.ones((scale,scale)))
        layer = layer.reshape(layer.shape[0], -1).transpose(1, 0)
        layer = layer.copy(order='C')
        temp_image = np.zeros((3, 64, 0), dtype="float32")
        temp_image = np.concatenate((temp_image, image), axis=2)

        #compare if pixels are in the regions that fullfill the ratio condition
        """img_mask = np.zeros((4096))
        for (lower,upper) in clusters:
            bool = np.where((layer > lower) & (layer < upper), 1, 0)
            img_mask = np.where(np.sum(bool, axis=1) > 0, 1, img_mask)
        img = copy.copy(image)
        img *= img_mask.reshape(1, 64, 64)
        temp_image = np.concatenate((temp_image, img), axis=2)
        img_mask = np.zeros((4096))
        distances, indices = faissknn_special.predict(layer, k=arr_C[special_indices].shape[0])
        for i in range(4096):
            if(np.sum(np.where(distances[i] < radius[indices[i]], 1, 0)) > 0):
                img_mask[i] = 1
        #img_mask = np.where(distances < radius[indices], 1, 0)

        img = copy.copy(image)
        img *= img_mask.reshape(1,64,64)
        temp_image = np.concatenate((temp_image, img), axis=2)"""

        distances, indices = faissknn.predict(layer, k=k_range)
        indices_b = np.sum(np.where(indices >= arr_A.shape[0], 1, 0), axis=1)
        ratio = np.divide(indices_b, k_range)
        for ratio_threshold in ratio_threshold_list:
            img_mask = np.where(ratio > ratio_threshold, 0, 1)
            img = copy.copy(image)
            img *= img_mask.reshape(1, 64, 64)
            temp_image = np.concatenate((temp_image, img), axis=2)
        temp_image = temp_image.transpose((1,2,0))
        img = Image.fromarray((temp_image * 255).astype("uint8"))
        draw = ImageDraw.Draw(img)
        # print(draw.textsize(str(1)))
        for i, value in enumerate(ratio_threshold_list):
            x, y = int((i + 1) * (img.width / (len(ratio_threshold_list) + 1))), 1
            draw.text((x, y), str(value), fill=(255, 255, 255))
        img.save(path + "/{:05d}.png".format(idx))

    if(output_video):
        out = cv2.VideoWriter(path + '/video.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24.0, (128,64))
        for step, x in enumerate(tqdm(valid_queue)):
            if (step < render_start_idx):
                continue
            if (step > render_stop_idx):
                break
            x = x[0] if len(x) > 1 else x
            x = x.cuda()
            x = utils.pre_process(x, args.num_x_bits)
            _, _, _, _, _, latent, pre_layers = model(x)
            image = x[0].detach().cpu().numpy()
            #if (kmeans_latent.predict(latent.detach().cpu().numpy().reshape(1, -1))[0] == max_critic_latent):
            layer = pre_layers[layer_idx][0].detach().cpu().numpy()
            scale = int(64 / layer.shape[1])
            layer = np.kron(layer, np.ones((scale, scale)))
            layer = layer.reshape(layer.shape[0], -1).transpose(1, 0)
            layer = layer.copy(order='C')
            _, indices = faissknn.predict(layer, k=k_range)
            indices_b = np.sum(np.where(indices >= arr_A.shape[0], 1, 0), axis=1)
            ratio = np.divide(indices_b, k_range)
            temp_image = np.zeros((3, 64, 0), dtype="float32")
            temp_image = np.concatenate((temp_image, image), axis=2)
            img = copy.copy(image)
            img_mask = np.where(ratio < ratio_threshold, 1, 0)
            img *= img_mask.reshape(1, 64, 64)
            temp_image = np.concatenate((temp_image, img), axis=2)
            temp_image = temp_image.transpose((1, 2, 0))
            img = Image.fromarray((temp_image * 255).astype("uint8"))
            numpy_image = np.array(img)
            opencv_image = cv2.cvtColor(numpy_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(opencv_image)
        cv2.destroyAllWindows()
        out.release()

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
    parser.add_argument('--image_number_clustering', default=5120, type=int)
    parser.add_argument('--render_start_idx', default=0, type=int)
    parser.add_argument('--render_stop_idx', default=50, type=int)
    parser.add_argument('--use_latent_filter', action='store_true', default=True)
    parser.add_argument('--load_kmeans_latent', action='store_true', default=True)
    parser.add_argument('--batch_size_trainset', type=int, default=8)
    parser.add_argument('--batch_size_testset', type=int, default=1)
    # experimental results
    parser.add_argument('--root', type=str, default='results',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='minecraft',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='minecraft',
                        choices=['minecraft'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='datasets/minecraft_lmdb',
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


