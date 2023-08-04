# Supplementary Material: Verifying Global Neural Network Specifications using Hyperproperties
 
This repository contains supplementary material for the [FoMLAS 2023](https://fomlas2023.wixsite.com/fomlas2023) paper [Verifying Global Neural Network Specifications using Hyperproperties](https://arxiv.org/abs/2306.12495).

## Dependency Fairness

The `dependency_fairness.py` script contains NNDHs for the Dependency Fairness hyperproperty.
It contains two equivalent variants of this NNDH. 
The first variant uses computations which are non-standard for neural networks, but which
are natural to use for encoding dependency fairness.
The second variant uses a more involved encoding which, however, only relies on
affine computations, max pooling, stacking and reshaping.

Concretely, `dependency_fairness.py` creates NNDH verification networks for
some input network. It tries to falsify dependency fairness using PGD.
Finally, it exports the two variants of the NNDH verification network as ONNX networks.

The `adult/` directory contains examples of such NNDH verification ONNX networks for three
networks trained on the Adult dataset. 
Refer to the `train_adult.py` script for how these networks were trained. 

