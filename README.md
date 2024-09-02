# TENNs-PLEIADES

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/building-temporal-kernels-with-orthogonal/gesture-recognition-on-dvs128-gesture)](https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture?p=building-temporal-kernels-with-orthogonal)

## Description

TENNs-PLEIADES is a (spatio)temporal convolutional network, where its temporal kernels are constructed by orthogonal polynomials. It is effective in capturing long-range temporal correlations, and is stable during training.

## Quickstart

First, install the necessary libraries in a working Python environment via `pip install -r requirements.txt`.

The `PleiadesLayer` can be used as a drop-in replacement for convolutional layers (only supporting `nn.Conv3d` layers for now), where the last dimension (assumed to be temporal) will be parameterized by orthogonal polynomials up to a given degree.

```python
from model import PleiadesLayer

layer = PleiadesLayer(2, 8, kernel_size=(3, 3, 20), degrees=4)
```

The structured temporal kernels can also easily be resampled into different kernel sizes without needing to retrain the network.

```python
layer.resample(10)  # downsample the kernel size from 20 to 10
```

## Citation

If you find TENNs-PLEIADES useful, please consider citing the [TENNs-PLEIADES: Building Temporal Kernels with Orthogonal Polynomials](https://arxiv.org/abs/2405.12179) paper:

```
@article{pei2024building,
  title={Building Temporal Kernels with Orthogonal Polynomials},
  author={Pei, Yan Ru and Coenen, Olivier},
  journal={arXiv preprint arXiv:2405.12179},
  year={2024}
}
```
