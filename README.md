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
pip install cvxopt
```

## Usage

```
python main.py --no-ssa
python main.py --ssa
```
- `--no-ssa` means that ssa is not used.
- `--ssa` means that ssa is used for monitoring and modifying the unsafe control.


## Acknowledgments
Part of the simulation environment code is coming from the course CS 7638: Artificial Intelligence for Robotics in GaTech. We get the permission from the lecturor Jay Summet to use this code for research.
