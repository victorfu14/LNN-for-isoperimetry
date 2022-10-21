# LNN for Isoperimetry

This repo utilize SOC structure (https://github.com/singlasahil14/SOC) to verify whether real world datasets satisfy isoperimety properties proposed by (https://arxiv.org/abs/2105.12806).


# Skew Orthogonal Convolutions

- **Skew Orthogonal Convolution (SOC)** is a convolution layer that has an orthogonal Jacobian and achieves state-of-the-art standard and provably robust accuracy compared to other orthogonal convolutions.
- **Last Layer normalization (LLN)** leads to improved performance when the number of classes is large.
- **Certificate Regularization (CR)** leads to significantly improved robustness of certificates.
- **Householder Activations (HH)** improve the performance of deeper networks.

## Prerequisites

- Python 3.7 or 3.8
- Pytorch 1.8
- einops. Can be installed using ```pip install einops```
- NVIDIA Apex. Can be installed using ```conda install -c conda-forge nvidia-apex```
- A recent NVIDIA GPU

## How to train 1-Lipschitz Convnets?

Real world datasets:
```python train.py --conv-layer CONV --activation ACT --block-size BLOCKS --dataset DATASET```

Synthetic distribution datasets, such as Gaussian distribution, add ```--synthetic``` flag:
```python train.py --conv-layer CONV --activation ACT --block-size BLOCKS --synthetic --dataset DATASET```

- CONV: bcop, cayley, soc
- ACT: maxmin, hh1, hh2 (hh1, hh2 are householder activations of order 1, 2).
- BLOCKS: 1, 2, 3, 4, 5, 6, 7, 8, represents the depth of neural network
- Use ```--lln``` to enable last layer normalization
- DATASET: cifar10/cifar100/gaussian/mnist/cifar5m.

For ```cifar5m```, the result is computed for a certain class. Please specify using ```--label LABEL``` where
LABEL is integers from 0 to 9

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

## Note

1. Given $X, X^\prime$, the loss is defined as $L(X, X^\prime)\coloneqq - \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - f(x^\prime_i))$
2. When reporting the loss when testing samples $X^{\prime \prime}, X^{\prime\prime\prime}$, we don't care about the sign, hence we report $\left\vert L(X^{\prime\prime}, X^{\prime\prime\prime}) \right\vert$.
3. Change the number of classes to $1$ since we only want to consider $\mathcal{F}$ such that $f\colon R^d \to \mathbb{R}$.
