# Skew Orthogonal Convolutions

+ **Skew Orthogonal Convolution (SOC)** is a convolution layer that has an orthogonal Jacobian and achieves state-of-the-art standard and provably robust accuracy compared to other orthogonal convolutions.
+ **Last Layer normalization (LLN)** leads to improved performance when the number of classes is large.
+ **Certificate Regularization (CR)** leads to significantly improved robustness of certificates.
+ **Householder Activations (HH)** improve the performance of deeper networks.

## Prerequisites

+ Python 3.7 or 3.8
+ Pytorch 1.8
+ requests. Can be installed using ```pip install requests```
+ einops. Can be installed using ```pip install einops```
+ NVIDIA Apex. Can be installed using ```conda install -c conda-forge nvidia-apex```
+ A recent NVIDIA GPU

> On GreatLakes, you need to manually download the datasets... `requests` doesn't really work.

## TODO

+ [ ] 🎨 Draw a line plot

## How to train 1-Lipschitz Convnets?

```python train_robust.py --conv-layer CONV --activation ACT --block-size BLOCKS --dataset DATASET --gamma GAMMA```

+ CONV: bcop, cayley, soc
+ ACT: maxmin, hh1, hh2 (hh1, hh2 are householder activations of order 1, 2; both illustrated below).
+ BLOCKS: 1, 2, 3, 4, 5, 6, 7, 8
+ GAMMA: certificate regularization coefficient
+ Use ```--lln``` to enable last layer normalization
+ DATASET: cifar10/cifar100.

## How to train Standard Convnets using Orthogonal Convolutions?

```python train_standard.py --conv-layer CONV --model-name MODEL --dataset DATASET```

+ CONV: bcop, cayley, soc (Use ```standard``` to train using standard convolution)
+ MODEL: resnet18, resnet34, resnet50, resnet101, resnet152
+ DATASET: cifar10, cifar100

## Citations

If you find this repository useful for your research, please cite:

```
@inproceedings{singlafeiziSOC2021,
  title={Skew Orthogonal Convolutions},
  author={Singla, Sahil and Feizi, Soheil},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021},
  pdf={http://proceedings.mlr.press/v139/singla21a/singla21a.pdf},
  url={https://proceedings.mlr.press/v139/singla21a.html}
}

@inproceedings{
  singla2022improved,
  title={Improved deterministic l2 robustness on {CIFAR}-10 and {CIFAR}-100},
  author={Sahil Singla and Surbhi Singla and Soheil Feizi},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=tD7eCtaSkR}
}
```
