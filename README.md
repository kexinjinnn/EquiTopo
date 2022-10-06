# Communication-Efficient Topologies for Decentralized Learning with $\mathcal{O}(1)$ Consensus Rate 

This code repository is for the paper
**Communication-Efficient Topologies for Decentralized Learning with $\mathcal{O}(1)$ Consensus Rate** to
be appeared in NeurIPS 2022. 

## Setup the environment

This code was trained and tested with

1. Python 3.7.5
2. PyTorch 1.4.0
3. Torchvision 0.5.0
4. tqdm 4.62.3
5. tensorboardX 2.4
6. [bluefog](https://github.com/Bluefog-Lib/bluefog) 0.3.0

You can also use the Bluefog
[docker image](https://bluefog-lib.github.io/bluefog/docker.html) for testing.

## Train a model with EquiTopo

We highly recommend downloading all data before training and putting them in a local folder. The following code is to run MNIST experiment with 17 decentralized network nodes using UEquiStatic (one-peer EquiTopo on a undirected graph) with M=4.

```bash
$ BLUEFOG_OPS_ON_CPU=1 bfrun -np 17 python train.py --topology UEquiStatic --M 4 --dataset mnist --atc-style --disable-dynamic-topology
```
## Test EquiTopo on synthetic data

We test the performances of EquiTopos using sythetic data to validate 
1) network-size-independent consensus rate (consensus.ipynb);
2) comparison with other commonly-used topologies in consensus rate (consensus.ipynb);
3) comparison with other commonly-used topologies in DSGD for strongly-convex problems (DSGD_convex.ipynb);
4) comparison with other commonly-used topologies in DSGT for non-convex problems (DSGT_nonconvex.ipynb).

## EquiTopo comparison on MNIST and CIFAR-10

We compare the performances of EquiTopos (D-EquiStatic, U-EquiStatic, OD-EquiDyn, OU-EquiDyn) on MINIST and CIFAR-10. You can find the script for testing in `run.sh`.

### Performance on MNIST and CIFAR-10

| Method  | MNIST Acc. | CIFAR-10 Acc. |
|--------|------|------|
| D-EquiStatic | 98.29% | 92.01% |
| U-EquiStatic | 98.26% | 91.74% |
| OD-EquiDyn | 98.39% | 91.44% |
| OU-EquiDyn | 98.12% | 91.56% |

## Citation

If you use this library or find the doumentation useful for your research, please consider citing:

Zhuoqing Song*, Weijian Li*, Kexin Jin*, Lei Shi, Ming Yan, Wotao Yin, and Kun Yuan,
**Communication-Efficient Topologies for Decentralized Learning with $\mathcal{O}(1)$ Consensus Rate**.
Advances in Neural Information Processing Systems (NeurIPS), 2022.

```txt
@inproceedings{Song_2022_EquiTopo,
    title = {Communication-Efficient Topologies for Decentralized Learning with $\mathcal{O}(1)$ Consensus Rate},
    author = {Zhuoqing Song* and Weijian Li* and Kexin Jin* and Lei Shi and Ming Yan and Wotao Yin and Kun Yuan},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = 2022
}
```

