# Safe and Sample efficient Reinforcement Learning for Clustered Dynamic Uncertain Environments

## Table of Contents
- [Introduction](#Introduction)
- [Install](#install)
- [Usage](#usage)
- [Acknowledgments](#Acknowledgments)

## Introduction
We provide code for evaluate the safety and sample-efficiency of our proposed RL framework.

For safety, we use Safe Set Algorithm (SSA).   
For efficiency, there are more strategies you can choose:  
1, Adapting SSA;  
2, Exploration (PSN, RND, None);  
3, Learning from SSA;  

The video result is shown below, agent is trained to drive to the goal while avoiding dynamic obstacles. The red means SSA is triggered.

<img src="docs/SSA_RL.gif" width="400" height="460">

## Install

```
conda create -n safe-rl
conda install python=3.7.9
pip install tensorflow==2.2.1
pip install future
pip install keras
pip install matplotlib
pip install gym
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
