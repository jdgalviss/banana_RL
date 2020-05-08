[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: imgs/scores.png "Scores"
[image3]: imgs/scores_visual.png "Scores Visual"
[image4]: imgs/visualDQN.gif "Visual agent"
[image5]: imgs/normalDQN.gif "Normal agent"


# Banana collector Agent: Navigation

## Introduction

In this project, an agent is trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies Installation
Follow the instructions from the Udacity's DRLND [repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system (This repo already contains the unity agents versions for Linux):
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the project folder, and unzip (or decompress) the file.


## Learning from pixels

It is also possible to train the agent so it learns directly from pixels (image seen by a camera located in the frontal part of the agent). In this case, the state is represented by a 84x84x3 image.


You need only select the environment that matches your operating system (Linux Version already contained in this repo):
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

## Instructions
### Navigate the environment
Follow the instructions in `Navigation.ipynb` to visualize the environment.

### Train the agent
Follow the `Train.ipynb` notebook to train an agent. Note that by changing the second cell's parameter `mode` you can choose between `visual` or `normal` operation so that the agent is trained either on 84x84x3 frontal images or a 37 state vector respectively. At the end, the agent will be trained until solving the task as shown in the following score vs. episodes plot:

Training on 37 elements vector:

![Scores][image2]

Training on pixels:

![ScoresVisual][image3]

### Test the agent
Follow the `Evaluate.ipynb` notebook to watch the intelligent agent. Note that by changing the second cell's parameter `mode` you can choose between `visual` or `normal` which will use the corresponding model and pretrained weights. The results can be seen on the following 2 animations:

Normal DQN Agent

![ScoresVisual][image5]

Visual DQN Agent

![ScoresVisual][image4]



## The Models
A Deep Q Network is used based on the approach proposed in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). Here, a Deep Neural Network "learns" the Q-value function, so that having a state vector as input (an image in the case of the visual environment), the network outputs a value for each one of the 4 actions while exploring the environment with an e-greedy policy. In order to improve the results, 2 of the techniques proposed on the paper are implemented:

1. Memory Replay: Several steps are stored on a memory buffer and then randomly sampled for training as defined in the `ReplayBuffer` class on the file  `model/dqn_agent.py`, hence coping with state-action correlation.

2. Fixed Q-Targets: This is done by means of a second network (same architecture as the DQN) that is used as a target network and updated every n steps.

### Visual Model
The model that is used when the state is represented by visual data (4x84x84x3 images) is defined in the file `model.py` using a succession of 3 convolutional layers, batch normalization, relu activations and 2 fully connected layers as follows (model parameters can be found in this [repo](https://github.com/yingweiy/drlnd_project1_navigation):

``` python
class VisualQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_units = 64, conv2_units = 64*2, conv3_units = 64*2, fc1_units=1152, fc2_units=64):
        super(VisualQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        s = 3
        self.conv1 = nn.Conv3d(3, conv1_units, kernel_size=(1, 3, 3), stride=(1,s,s))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2))
        self.bn1 = nn.BatchNorm3d(conv1_units)
        self.conv2 = nn.Conv3d(conv1_units, conv2_units, kernel_size=(1, 3, 3), stride=(1,s,s))
        self.bn2 = nn.BatchNorm3d(conv2_units)
        self.conv3 = nn.Conv3d(conv2_units, conv3_units, kernel_size=(4, 3, 3), stride=(1,s,s))
        self.bn3 = nn.BatchNorm3d(conv3_units)
        self.fc1 = nn.Linear(fc1_units, fc2_units)
        self.fc2 = nn.Linear(fc2_units, action_size)
```

It is important to note that the state is given by the 4 most recent images seen by the agent, this is done to give the dqn agent temporal information. This feature is defined in the `Environment` class in the `environment.py` file.

### State Vector Model
Since this is a more summarized data representation, this model is simpler and only consists of a succession of 3 fully connected layers and relu activations as follows:

``` python
class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
```

## Future Work
Several techniques have been implemented to improve the performance of DeepQ-Learning, some of which could be implemented for this particular case are:
* [Double DQN](https://arxiv.org/abs/1509.06461): In order to avoid over estimation of Q-values by having one network to select the best action from the target function and another one to evaluate it. This helps us to avoid the propagation of accidental high rewards. For example, we could use the network used in the fixed-q targets approach to evaluate actions.
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Instead of sampling randomly from the replay buffer, this approach suggests to take into account that some experiences might be more valueble (important or scarse). To define how important one experience is, we could store the TD Error for each step on the Replay Buffer and then sample based on its value.
* [Dueling Networks](https://arxiv.org/abs/1511.06581): This technique proposes a new arquitecture where instead of only estimating the state-action value function Q(s,a), we estimate the state value function V(s) and advantage values A(s,a). Then the desired Q-Values are obtained by combaining V(s) and A(s,a). The intuition behind this is that the value of states don't vary a lot accross actions.