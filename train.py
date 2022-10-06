# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function


from bluefog.common import topology_util
import bluefog.torch as bf
import argparse
import os
import sys
import math
import warnings
import resnet
warnings.simplefilter('ignore')
import numpy as np
import networkx as nx
from itertools import chain
from typing import List, Tuple, Dict, Iterator, Optional
import collections
import time
from EquiStatic import *
from EquiDyn import *

from scipy.io import savemat
import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import tensorboardX
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

cwd_folder_loc = os.path.dirname(os.path.abspath(__file__))
# Training settings
parser = argparse.ArgumentParser(
    description="PyTorch ImageNet Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='The dataset to train with.')

parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=130,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay')

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument("--seed", type=int, default=100, help="random seed")
parser.add_argument('--dist-optimizer', type=str, default='neighbor_allreduce',
                    help='The type of distributed optimizer. Supporting options are [win_put, ' +
                    'neighbor_allreduce, hierarchical_neighbor_allreduce, allreduce, ' +
                    'gradient_allreduce, horovod]')
parser.add_argument('--atc-style', action='store_true', default=False,
                    help='If True, the step of optimizer happened before communication')
parser.add_argument('--disable-dynamic-topology', action='store_true',
                    default=False, help=('Disable each iteration to transmit one neighbor ' +
                                         'per iteration dynamically.'))

parser.add_argument('--topology', type=str, default='exp',
                    help='The topology used in decentralized algorithms to connect all nodes. Supporting options are [DEquiStatic, ' +
                    'UEquiStatic, ring, centralizedSGD]')
parser.add_argument('--M', type=int, default=1,
                    help='parameter for EquiTopos')
parser.add_argument('--complete', action='store_true', default=False,
                    help='If True, generating EquiDyn using M=n-1.')
parser.add_argument('--eta', type=float, default=1.0,
                    help='The weights in EquiDyn graphs, 0< eta <=1')
parser.add_argument('--equiseed', type=int, default=106,
                    help='An integer used as the random seed generate EquiStatics')
parser.add_argument('--equirng', type=int, default=42,
                    help='An integer used as the random seed generate EquiDyns')
                

args = parser.parse_args()

# for mnist only 
if args.dataset == 'mnist':
    args.batch_size = 64
    args.momentum = 0.5
    args.wd = 0.0
    args.val_batch_size = 1000
    # we do not use adaptive lr
    args.base_lr = 0.01
    args.epochs = 20

args.cuda = (not args.no_cuda) and (torch.cuda.is_available())
allreduce_batch_size = args.batch_size * args.batches_per_allreduce


# Bluefog: initialize library.
bf.init()


# Set up topology for bluefog
if args.topology == 'DEquiStatic':
    if not args.complete:
        W, Ms= DEquiStatic(bf.size(), seed=args.equiseed, M=args.M)
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        bf.set_topology(G, is_weighted=True)
elif args.topology == 'UEquiStatic':
    if not args.complete:
        W, Ms= UEquiStatic(bf.size(), seed=args.equiseed, M=args.M)
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        bf.set_topology(G, is_weighted=True)
elif args.topology == 'centralizedSGD':
    args.dist_optimizer = 'gradient_allreduce'
elif args.topology == 'ring':
    G = topology_util.RingGraph(bf.size())
    bf.set_topology(G)


torch.manual_seed(args.seed)

if args.cuda:
    print("using cuda.")
    # Bluefog: pin GPU to local rank.
    device_id = bf.local_rank() % torch.cuda.device_count() 
    torch.cuda.set_device(int(device_id))
    torch.cuda.manual_seed(args.seed)
else:
    print("using cpu")

cudnn.benchmark = True

# Bluefog: print logs on the first worker.
verbose = 1 if bf.rank() == 0 else 0

# Bluefog: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(
    args.log_dir) if bf.rank() == 0 else None


kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
if args.dataset == "cifar10":
    train_dataset = datasets.CIFAR10(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [   transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "mnist":
    train_dataset = datasets.MNIST(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [   transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
else:
    raise ValueError("Args dataset should be either cifar10 or mnist")

# Bluefog: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bf.size()` and `rank=bf.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bf.size(), rank=bf.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs
)

if args.dataset == "cifar10":
    val_dataset = datasets.CIFAR10(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "mnist":
    val_dataset = datasets.MNIST(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=False,
        download=False,
        transform=transforms.Compose(
            [   transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bf.size(), rank=bf.rank()
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs
)

# this model is for mnist only!!
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


if args.dataset == "cifar10":
    model = resnet.__dict__["resnet20"]()
elif args.dataset == "mnist":
    model = Net()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Bluefog: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(
    model.parameters(),
    lr=(args.base_lr * args.batches_per_allreduce * bf.size()),
    momentum=args.momentum,
    weight_decay=args.wd,
)

# Bluefog: wrap optimizer with DistributedOptimizer.
if args.dist_optimizer != 'horovod':
    base_dist_optimizer = (
        bf.DistributedAdaptThenCombineOptimizer if args.atc_style else
        bf.DistributedAdaptWithCombineOptimizer)
if args.dist_optimizer == 'win_put':
    optimizer = bf.DistributedWinPutOptimizer(optimizer, model=model)
elif args.dist_optimizer == 'allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.allreduce)
elif args.dist_optimizer == 'neighbor_allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce)
elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model,
        communication_type=bf.CommunicationType.hierarchical_neighbor_allreduce)
elif args.dist_optimizer == 'gradient_allreduce':
    optimizer = bf.DistributedGradientAllreduceOptimizer(
        optimizer, model=model)
elif args.dist_optimizer == 'horovod':
    optimizer = bf.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())
else:
    raise ValueError('Unknown args.dist-optimizer type -- ' + args.dist_optimizer + '\n' +
                     'Please set the argument to be one of ' +
                     '[neighbor_allreduce, gradient_allreduce, allreduce, ' +
                     'hierarchical_neighbor_allreduce, win_put, horovod]')

# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    train_accuracy = Metric("train_accuracy")

    with tqdm(total=len(train_loader), desc="Train Epoch     #{}".format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if not (args.dataset == "mnist"):
                adjust_learning_rate(epoch, batch_idx)
            if not args.disable_dynamic_topology:
                dynamic_topology_update(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i: i + args.batch_size]
                target_batch = target[i: i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                if args.dataset == "mnist":
                    loss = F.nll_loss(output, target)
                else: 
                    loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "accuracy": 100.0 * train_accuracy.avg.item(),
                }
            )
            t.update(1)


    if log_writer:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)
        log_writer.add_scalar("train/accuracy", train_accuracy.avg, epoch)
    return train_loss.avg, train_accuracy.avg

def validate(epoch, record):
    model.eval()
    val_loss = Metric("val_loss")
    val_accuracy = Metric("val_accuracy")

    with tqdm(total=len(val_loader), desc="Validate Epoch  #{}".format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "accuracy": 100.0 * val_accuracy.avg.item(),
                    }
                )
                t.update(1)

    if log_writer:
        log_writer.add_scalar("val/loss", val_loss.avg, epoch)
        log_writer.add_scalar("val/accuracy", val_accuracy.avg, epoch)
    record.append((val_loss.avg, val_accuracy.avg*100))
    return val_loss.avg, val_accuracy.avg



# Bluefog: using `lr = base_lr * bf.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bf.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1.0 / bf.size() * (epoch * (bf.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 50:
        lr_adj = 0.1
    elif epoch < 100:
        lr_adj = 1e-2
    elif epoch < 120:
        lr_adj = 1e-3
    else:
        lr_adj = 1e-4
    for param_group in optimizer.param_groups:
        param_group["lr"] = (
            args.base_lr * bf.size() * args.batches_per_allreduce * lr_adj
        )


if not args.disable_dynamic_topology and (args.dist_optimizer != 'horovod'):
    if args.dist_optimizer == 'neighbor_allreduce':
        if args.topology == 'UEquiStatic':
            if args.complete:
                dynamic_neighbor_allreduce_gen = GetOUEquiDynCompleteSendRecvRanks(
                    bf.load_topology(), bf.rank(), seed=args.equirng)
            else:
                dynamic_neighbor_allreduce_gen = GetOUEquiDynSendRecvRanks(
                    bf.load_topology(), bf.rank(), seed=args.equirng, As=Ms)
        elif args.topology == 'DEquiStatic':
            if args.complete:
                dynamic_neighbor_allreduce_gen = GetODEquiDynCompleteSendRecvRanks(
                    bf.load_topology(), bf.rank(), seed=args.equirng)
            else:
                dynamic_neighbor_allreduce_gen = GetODEquiDynSendRecvRanks(
                    bf.load_topology(), bf.rank(), seed=args.equirng, As=Ms)
        elif args.topology == 'exp':
            if bf.is_homogeneous() and bf.size() > bf.local_size():
                dynamic_neighbor_allreduce_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
                    bf.size(),
                    local_size=bf.local_size(),
                    self_rank=bf.rank())
            else:
                dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
                    bf.load_topology(), bf.rank())
        else:
            raise Exception("this option only works for EquiTopo or Exponential graph")
    elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
        # This optimizer can use following dynamic topo only so far.
        dynamic_machine_neighbor_allreduce_gen = topology_util.GetExp2DynamicSendRecvMachineRanks(
            world_size=bf.size(),
            local_size=bf.local_size(),
            self_rank=bf.rank(),
            local_rank=bf.local_rank()
        )
    else:
        dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
            bf.load_topology(), bf.rank())

def dynamic_topology_update(epoch, batch_idx):
    if args.dist_optimizer == 'win_put':
        if epoch < 3:
            return
        num_out_neighbors = len(bf.out_neighbor_ranks())
        sent_neighbor = bf.out_neighbor_ranks()[batch_idx % num_out_neighbors]
        optimizer.dst_weights = {sent_neighbor: 1.0}
    elif args.dist_optimizer == 'neighbor_allreduce':
        if args.topology == 'DEquiStatic':
            send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
            optimizer.dst_weights = send_neighbors
            optimizer.src_weights = {r: args.eta * (bf.size() - 1) / bf.size() for r in recv_neighbors}
            optimizer.self_weight = 1 - args.eta + args.eta / bf.size()
            optimizer.enable_topo_check = True
        elif args.topology == 'UEquiStatic':
            send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
            if send_neighbors == None:
                optimizer.dst_weights = [bf.rank()]
                optimizer.src_weights = {bf.rank(): 0.0}
                optimizer.self_weight = 1.0
                optimizer.enable_topo_check = True
            else:
                optimizer.dst_weights = send_neighbors
                optimizer.src_weights = {r: args.eta * (bf.size() - 1) / bf.size() for r in recv_neighbors}
                optimizer.self_weight = 1 - args.eta + args.eta / bf.size()
                optimizer.enable_topo_check = True
        else:
            send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
            optimizer.dst_weights = send_neighbors
            optimizer.src_weights = {r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
            optimizer.self_weight = 1 / (len(recv_neighbors) + 1)
            optimizer.enable_topo_check = False
    elif args.dist_optimizer == 'hierarchical_neighbor_allreduce':
        send_machines, recv_machines = next(dynamic_machine_neighbor_allreduce_gen)
        optimizer.dst_machine_weights = send_machines
        optimizer.src_machine_weights = {r: 1/(len(recv_machines) + 1) for r in recv_machines}
        optimizer.self_weight = 1 / (len(recv_machines) + 1)
        optimizer.enable_topo_check = False
    else:
        pass


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val):
        self.sum += bf.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == "__main__":
    
    timearray=time.localtime(float(time.time()))
    tt=time.strftime('%Y-%m-%d-%H-%M-%S',timearray)
    counter = [0]

    test_record = []
    data_dict = collections.defaultdict(list)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch, test_record)

        data_dict['train_loss'].append(train_loss)
        data_dict['train_accuracy'].append(train_acc)
        data_dict['val_loss'].append(val_loss)
        data_dict['val_accuracy'].append(val_acc)
    
    bf.barrier()


    if bf.rank() == 0:

        # save the experiment data
        print()

        fname = "test"
        savemat(fname + ".mat", data_dict)

        print("Results saved! :)")
