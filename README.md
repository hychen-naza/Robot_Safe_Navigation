# Robot Safe Navigation in Clustered Dynamic Environments

## Table of Contents
- [Introduction](#Introduction)
- [Install](#install)
- [Usage](#usage)
- [Acknowledgments](#Acknowledgments)

## Introduction
We provide code for evaluate the safety of our proposed safe control method - Safe Set Algorithm (SSA).

<img src="docs/SSA_RL.gif" width="400" height="460">

## Install

```
conda create -n safe_nav
conda install python=3.9
pip install cvxopt
```

## Usage

```
python train.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --no-ssa-buffer
python train.py --display {none, turtle} --explore {none, psn, rnd} --qp --no-ssa-buffer
python train.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --ssa-buffer
```
- `--display` can be either `none` or `turtle` (visulization).
- `--explore` specifies the exploration strategy that the robot uses. 
- `--no-qp` means that we use vanilla SSA.
- `--qp` means that we use adapted SSA.
- `--no-ssa-buffer` means that we use the default learning.
- `--ssa-buffer` means that we use the safe learning from SSA demonstrations.


## Acknowledgments
Part of the simulation environment code is coming from the course CS 7638: Artificial Intelligence for Robotics in GaTech. We get the permission from the lecturor Jay Summet to use this code for research.
