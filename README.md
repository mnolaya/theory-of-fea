# MECH.5130: Theory of Finite Element Analysis

**Michael N. OLaya**

---

## Introduction

Welcome to Michael's Finite Element (`mfe`) solver package written exclusively in Python using a object-oriented approach and is heavily vectorized where possible. This library was developed primarily for Dr. Stapleton's MECH.5130: Theory of Finite Element Analysis course at University of Massachusetts Lowell.

## Requirements

1. Python 3.10+

## Package installation

First, download and unzip or clone the repo to your local machine. If cloning, the command is:

```bash
git clone https://github.com/mnolaya/theory-of-fea
```

It is recommended to create a Python virtual environment to install the package into. If using [conda](https://docs.anaconda.com/miniconda/):

```bash
conda create -n some_package_name python=3.10
```

Then, navigate to the top level of the repo where this README is located, activate your newly created virtual environment, and install the package, which should include all necessary dependencies.

```bash
cd theory-of-fea
conda activate some_package_name
pip install .
```

## Homeworks

Homework solutions have been organized into a series of Jupyter Notebooks for convenient review of derivations and figures and to demonstrate the workings of the `mfe` package with code snippets. Follow the links below, or check the individual folders under the [homeworks](./homeworks/) directory.

- [Homework 1](./homeworks/hw1/hw1_soln.ipynb)
- [Homework 2](./homeworks/hw2/hw2_soln.ipynb)
- [Homework 3](./homeworks/hw3/hw3_soln.ipynb)
- [Homework 4](./homeworks/hw4/hw4_soln.ipynb)
- [Homework 5](./homeworks/hw5/hw5_soln.ipynb)
- [Homework 6](./homeworks/hw6/hw6_soln.ipynb)
- [Homework 7](./homeworks/hw7/hw7_soln.ipynb)
- [Homework 8](./homeworks/hw8/hw8_soln.ipynb)

## Source code

The source code can be found for review in various `.py` files [here](./mfe). Note that there are some legacy functions that either need to be renamed, removed, or cleaned up and re-implemented. Work in progress!