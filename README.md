# DMALAX - Discrete Metropolis-Adjusted Langevin Algorithm in JAX

This repository contains  the code for the [JAX](https://github.com/google/jax) based implementation of the work by __Zhang et.al 2022 titled [A Langevin-like Sampler for Discrete Distributions](https://arxiv.org/pdf/2206.09914.pdf)__. It's design is heavily insipred by [blackjax](https://github.com/blackjax-devs/blackjax/) even borrowing some code from the api. (_I implemented this code in a separate repo as part of learning how samplers for discrete distributions work, and I plan to send a PR to the official blackjax repo :crossed_fingers:_)

## Usage

Please check the notebooks in `examples` directory for how to use the kernel 


## Todo

- Extend the kernel for `Categorical` distributions. Currently only binary-valued distributions are supported
- Add more example notebooks that implement:
  - Potts Model
  - Restricted-Boltzmann Machine (RBM) Model
  - Bayesian Neural Network (BNN)
