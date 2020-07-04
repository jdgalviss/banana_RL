
"""
Training loop.
"""
import numpy as np
import torch
from collections import deque
from tensorboardX import SummaryWriter

def train(env, agent, mode, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, goal_score = 13.0):
    """ Training our Deep Q-Learning agent.
    Params
    ======
        env: Unity environment
        agent: implementation of the dqn agent
        mode: normal(state vector of size 37) or visual (state consists of images seen by the agent).
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    writer = SummaryWriter("runs")
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps) # Calculate agent from Deep Q-Network
            next_state, reward, done, _ = env.step(action) # Perform action and get new state and reward
            agent.step(state, action, reward, next_state, done) # Perform agent step (save experience and learn every n steps)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        writer.add_scalar('Current_score', score, i_episode)
        writer.add_scalar('AverageScore', np.mean(scores_window), i_episode)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=goal_score:  # If environment, solved, save network weights
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/checkpoint-{}.pth'.format(mode))
            np.savez("scores-{}.npz".format(mode),np.array(scores))
            break


def evaluate(env, agent, mode):
    # Load model
    checkpoint_path = 'checkpoints/checkpoint-{}.pth'.format(mode)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    for _ in range(3):
        state = env.reset()                                # get the current state
        score = 0                                          # initialize the score
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action) # Perform action and get new state and reward
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        print("Score: {}".format(score)) 
