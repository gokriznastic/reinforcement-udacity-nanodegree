[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Environment and the task:

In the given 2 dimensional environment, there are two types of bananas, yellow ones and blue ones. If the agent catches yellow one, he gets positive reward (+1). But if it catches blue one, he gets negative reward (-1). Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

We need to implement a DeepRL agent using Python and Pytorch.

The simulation contains a single agent that navigates a large environment. At each time step, it has four actions at its disposal:

- `0` - go forward
- `1` - go backward
- `2` - turn left
- `3` - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### The approach and solution

In this solution we have used a simple **Deep Q-learning** agent implemented using a simple feed-forward network in Pytorch. More details are provided in [project report](PROJECT_REPORT.md) along with the [weights](checkpoint.pth) for our trained agent. 

### Setting up the project

1. Follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.
2. Download the Unity Environment from one of the links below. You need only select the environment that matches your operating system:
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
3. Then, place the file in the `p1_navigation/` folder in this repository, and unzip (or decompress) the file (already provided here).
4. After you have followed the instructions above, open `Navigation.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

### Working with the project

To train the agent, start jupyter notebook, open `Navigation.ipynb` and execute! For more information, please check instructions inside the notebook.
For the network used for Q-learning take a look at `model.py` whereas to look into the implementation of the DeepRL agent and Q-learning algorithm look into `dqn_agent.py`.
