# MMNet

Pytorch implementation for MMNet: A Multi-Scale Multimodal Model for
End-to-End Grouping of Fragmented UI Elements.

### Requirements

```
pip install -r requirements.txt
```

### Usage

This is the Pytorch implementation of MMNet. It has been trained and tested on Linux (Ubuntu20 + Cuda 11.6 + Python 3.9 + Pytorch 1.13 + NVIDIA GeForce RTX 3090 GPU), and it can also work on Windows.

### Getting Started 

```
git clone https://github.com/ssea-lab/MMNet
cd MMnet
```

### Train Our Model

* Start to train with

  ```
  torchrun --nnodes 1 --nproc_per_node 1  main.py --batch_size 10 --lr 5e-4Test Our Model
  ```

### Test Our Model

* Start to test with

  ```
  torchrun --nnodes 1 --nproc_per_node 1  main_ddp.py --evaluate --resume ./work_dir/set-wei-05-0849/checkpoints/latest.pth --batch_size 40
  ```

### Baselines of UI Fragmented Element Classification

#### EfficientNet

* Start to train with

  ```
  torchrun --nnodes 1 --nproc_per_node 1  efficient_main.py --batch_size 4 --lr 5e-4
  ```

* Start to test with 

  ```
  torchrun --nnodes 1 --nproc_per_node 4  efficient_main.py --evaluate --resume ./work_dir/efficient_net/latest.pth --batch_size 8
  ```

#### Vision Transformer(ViT)

- Start to train with

```
torchrun --nnodes 1 --nproc_per_node 4  vit_main.py --batch_size 4 --lr 5e-4
```

- Start to test with

```
torchrun --nnodes 1 --nproc_per_node 4  vit_main.py --evaluate --resume ./work_dir/vit/latest.pth --batch_size 8
```

#### Swin Transformer

- Start to train with

```
torchrun --nnodes 1 --nproc_per_node 4  sw_vit_main.py --batch_size 4 --lr 5e-4
```

- Start to test with

```
torchrun --nnodes 1 --nproc_per_node 4  sw_vit_main.py --evaluate --resume ./work_dir/swin/latest.pth --batch_size 8
```

### ACKNOWNLEDGES

The implementations of EfficientNet, Vision Transformer, and Swin Transformer are based on the following GitHub Repositories. Thank for the works.

- EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch
- ViT: https://github.com/lucidrains/vit-pytorch
- Swin Transformer: https://github.com/microsoft/Swin-Transformer
