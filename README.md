# Parl-CarRacing-V0
环境:
## Install Requirements

```shell
# or try: pip install -r requirements.txt
pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
```
使用paddle的parl的算法，对gym的CarRacing-V0进行强化学习训练

## PARL Basics

[PARL](https://github.com/PaddlePaddle/PARL) is a flexible and high-efficient reinforcement learning framework. PARL aims to build an agent for training algorithms to perform complex tasks. The main abstractions introduced by PARL that are used to build an agent recursively are the following:

- **Model** is abstracted to construct the forward network which defines a policy network or critic network given state as input.

- **Algorithm** describes the mechanism to update parameters in Model and often contains at least one model.

- **Agent**, a data bridge between the environment and the algorithm, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.
效果如下:
![image](https://github.com/Attackzzw/Parl-CarRacing-V0/blob/master/carRacing.gif)

![image](https://github.com/Attackzzw/Parl-CarRacing-V0/blob/master/1.png)
