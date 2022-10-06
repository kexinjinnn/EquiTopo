#!/bin/bash

# Run deep learning experiments with MNIST:

# static Exp.:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology exp --dataset mnist --atc-style --disable-dynamic-topology

# O.-P. Exp.:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology exp --dataset mnist --atc-style 

# ring:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology ring --dataset mnist --atc-style --disable-dynamic-topology

# centralized SGD:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology centralizedSGD --dataset mnist --atc-style --disable-dynamic-topology

# D-EquiStatic:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology DEquiStatic --M 4 --dataset mnist --atc-style --disable-dynamic-topology

# U-EquiStatic:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology UEquiStatic --M 4 --dataset mnist --atc-style --disable-dynamic-topology

# OD-EquiDyn:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology DEquiStatic --dataset mnist --atc-style --complete --eta 0.53 

# OU-EquiDyn:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology UEquiStatic --dataset mnist --atc-style --complete --eta 0.53 

# Run deep learning experiments with CIFAR-10:

# static Exp.:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology exp --atc-style --disable-dynamic-topology

# O.-P. Exp:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology exp --atc-style 

# ring:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology ring --atc-style --disable-dynamic-topology

# centralized SGD:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology centralizedSGD --atc-style --disable-dynamic-topology

# D-EquiStatic:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology DEquiStatic --M 5 --atc-style --disable-dynamic-topology 

# U-EquiStatic:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology UEquiStatic --M 5 --equiseed 126 --atc-style --disable-dynamic-topology

# OD-EquiDyn:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology DEquiStatic --atc-style --complete --eta 0.53 

# OU-EquiDyn:
BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology UEquiStatic --atc-style --complete --eta 0.53 --equirng 200 
