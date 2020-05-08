[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: imgs/scores.png "Scores"
[image3]: imgs/scores_visual.png "Scores Visual"
[image4]: imgs/visualDQN.gif "Visual agent"
[image5]: imgs/normalDQN.gif "Normal agent"


# Banana collector Agent: Navigation

## The environment
![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Learning Algorithm
In this project a deep-Q Agent is trained to complete the task. A Deep Neural Network is used to approximate the Q-value function (Q(s,a)) that is used to find an optimal policy.

### Training
On each training step the deep Q-Agent :

1. Computes the action values for the current states and selects an action based on an **epsilon-greedy** policy.
2. Performs a step based on the selected action, obtaining the next state and collecting reward.
3. Updates the deep neural network parameters by performing a learning step which consists on:
    1. Storing the step data (state, action, reward, next_state) in a Replay buffer, following the **Memory Replay** technique described in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    2. Every n steps, sample a batch size from the Replay buffer and train the neural network.
        * For this, a second neural network is used as a target, this network is only updated using a **soft update**, defined by the parameter tau. (This part is based on the fixed q-targets technique described in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)).
        * The loss function is defined as `loss = F.mse_loss(Q_expected, Q_targets)`
        * Back propagation is performed and the network is trained using **Adam** as optimizer.
4. Current state is updated and the process is repeated.

The critical parameters for the training of the deep-Agent are:

- **`eps_start = 1.0`** - Initial value for the epsilon greedy policy (i.e. at the beginning of the training, actions will be selected randomly).
- **`eps_end = 0.01`** - Minimum value that the epsilon value can reach so that even at the last stages of the training, some level of exploration is performed on the action-state space.
- **`eps_decay=0.995`** - For each episode, the value will decay with the formula (eps = max(eps_end, eps_decay*eps)).
- **`buffer_size = 1e5`** - Size of the Replay buffer where step data (state, action, reward, next_state) is stored
- **`batch_size = 64`** - Size of the batch of data extracted from the Replay Buffer for each learning step.
- **`gamma = 0.99`** - Parameter used to prioritize rewards closer to the present.
- **`tau = 1e-3`** - Parameter used for the soft update of the target network.
- **`lr = 5e-4`** - Learning rate parameter for the Adam Optimizer.
- **`update_every = 4`** - Defines the number of exploration steps performed per learning step.


### Training Results
After Training the agent, it solved the environment in about 450 episodes as shown in the following figure:

![Scores][image2]

The agent was also trained, using the 3x84x84 image seen by the agent as state, the environment was solved in about 970 episodes. It's important to note that for this, sets of 4 images were used to define the state in order to give a temporal sense to the data:

![ScoresVisual][image3]

At the end, the trained agents perform as shown on the following gifs:

Normal DQN Agent

![ScoresVisual][image5]

Visual DQN Agent

![ScoresVisual][image4]


## Future Work
Several techniques have been implemented to improve the performance of DeepQ-Learning, some of which could be implemented for this particular case are:
* [Double DQN](https://arxiv.org/abs/1509.06461): In order to avoid over estimation of Q-values by having one network to select the best action from the target function and another one to evaluate it. This helps us to avoid the propagation of accidental high rewards. For example, we could use the network used in the fixed-q targets approach to evaluate actions.
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Instead of sampling randomly from the replay buffer, this approach suggests to take into account that some experiences might be more valueble (important or scarse). To define how important one experience is, we could store the TD Error for each step on the Replay Buffer and then sample based on its value.
* [Dueling Networks](https://arxiv.org/abs/1511.06581): This technique proposes a new arquitecture where instead of only estimating the state-action value function Q(s,a), we estimate the state value function V(s) and advantage values A(s,a). Then the desired Q-Values are obtained by combaining V(s) and A(s,a). The intuition behind this is that the value of states don't vary a lot accross actions.