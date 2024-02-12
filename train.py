# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, \
    over_write_args_from_file, TBLog


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')

    '''
    Training Configuration of FixMatch
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=5,
                        help='logging frequencu')
    parser.add_argument('-nl', '--num_labels', type=int, default=400)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True',
                        help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None,
                        help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    ## nlp dataset arguments 
    parser.add_argument('--max_length', type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args


def main(args):



    # Print Python version
    print("### Python Version:", sys.version)

    # Check if CUDA is available in the current PyTorch installation
    if torch.cuda.is_available():
        # Print CUDA version as reported by PyTorch
        print("######## Torch Version:", torch.__version__)
        print("CUDA Version:", torch.version.cuda)
        # Additionally, print the number of CUDA devices detected
        print("Number of CUDA Devices:", torch.cuda.device_count())
        # Print the name of the first CUDA device, if available
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Check your PyTorch installation and GPU drivers.")





    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    print(f"CONFIG FILE : {args.c}")
    nb_classes_dict = {'cifar10': 10, 'cifar100': 100, 'stl10': 10, 'svhn': 10, 'imagenet': 1000}
    print(f"DATASET : {args.dataset}")
    args.num_classes = nb_classes_dict[args.dataset]
    assert args.num_train_iter % args.epoch == 0, \
        f"NB OF TOTAL training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    # set appropriate wandb name
    net_print_dic = {'wrn_28_2': 'wrn_28_2', 'wrn_28_2_proto': 'wrn_28_2', 'wrn_28_8': 'wrn_28_8',
                     'wrn_28_8_proto': 'wrn_28_8',
                     'wrn_var_37_2': 'wrn_var_37_2', 'wrn_var_37_2_proto': 'wrn_var_37_2',
                     'vit_small_patch2_32': 'vit_small_patch2_32', 'vit_small_patch2_32_proto': 'vit_small_patch2_32',
                     'vit_tiny_patch2_32': 'vit_tiny_patch2_32', 'vit_tiny_patch2_32_proto': 'vit_tiny_patch2_32',
                     'vit_small_patch16_224': 'vit_small_patch16_224',
                     'vit_small_patch16_224_proto': 'vit_small_patch16_224',
                     'vit_base_patch16_96': 'vit_base_patch16_96', 'vit_base_patch16_96_proto': 'vit_base_patch16_96',
                     'vit_base_patch16_224': 'vit_base_patch16_224',
                     'vit_base_patch16_224_proto': 'vit_base_patch16_224'

                     }

    args.wandb_project = getattr(args, 'wandb_project', f"semisupcon_{args.dataset}{net_print_dic[args.net]}")
    # args.wandb_project = f"semisupcon_{args.dataset}_{args.num_labels}_{net_print_dic[args.net]}"
    if args.algorithm in ['fixmatch', 'flexmatch']:
        algo_print = args.algorithm
        loss_print = ''
        pl_print = ''
    elif args.algorithm in ["semisupcon", "semisupconproto","semisupconproto2","flexmatch_contrastive"]:
        loss_print = f'_{args.loss}_tau={args.p_cutoff}'
        if args.algorithm == "semisupconproto" and 'Weights' in args.loss:
            loss_print += f'_lambdaProto={args.lambda_proto}'
        if args.algorithm in ["semisupcon","semisupconproto","semisupconproto2","flexmatch_contrastive"]:
            algo_print = args.algorithm
        if args.pl == 'pl=max':
            pl_print = 'max_'
        elif args.pl == 'softmax':
            pl_print = f'pl=softmaxT={args.pl_temp}_'
        else:
            raise ValueError(f"Unknown pl {args.pl}")
    elif args.algorithm == "unsup":
        algo_print = args.algorithm
        loss_print = args.loss
        pl_print = ''

    print_prefix_wandb_name = '' if getattr(args, 'wandb_name', None) is None else f'{args.wandb_name}_'
    args.save_name_wandb = str(
        f'{print_prefix_wandb_name}{algo_print}{loss_print}_{pl_print}bs{args.batch_size}_lr{args.lr}_seed{args.seed}')
    if 'OAR_JOB_ID' in os.environ:
        args.save_name_wandb += f"_AK{os.environ['OAR_JOB_ID']}"
    if 'SLURM_JOB_ID' in os.environ:
        args.save_name_wandb += f"_JZ{os.environ['SLURM_JOB_ID']}"
        args.data_dir = "/gpfsscratch/rech/cgs/ued97kp/semisupcon/data" #PERSO TO CLEAN


    # set savename
    # args.save_name should not be set to none for run that will requires to resume weights .
    # otherwise the model is save using the wandb unique name
    if args.save_name is None:
        # args.save_name = f"{args.algorithm}_{args.dataset}_{args.num_labels}_{args.seed}" # savename is used for saving model dirname (we dont need as much details as wandb name)
        args.save_name = args.save_name_wandb
    # set save_path
    print(f" #### SAVE PATH : {args.save_dir}/{args.save_name}")
    print(f" #### SAVE NAME WANDB : {args.save_name_wandb}")

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))

    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if args.load_path == 'auto':
            args.load_path = os.path.join(args.save_dir, args.save_name, 'latest_model.pth')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument.')
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size
        print(f'world_size: {args.world_size},ngpus_per_node: {ngpus_per_node}')

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')


    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path

    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info(f"#### Resume load path {format(args.load_path)} does not exist ####")

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    if hasattr(model, 'results_dict'):
        for key, item in model.results_dict.items():
            logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)
