#!/usr/bin/env python

import argparse
from environment import BananaEnvironment
from dqn_agent import Agent
from training import train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="path to saved model", type=str, default=None)
parser.add_argument("--eval", help="run in evaluation mode.  no training.  no noise.", default=False)
parser.add_argument("--mode", help="run in visual(state vector corresponds to image seen by the agent) or normal mode", default='normal')

args = parser.parse_args()

# create environment and agent, then run training
environment = BananaEnvironment(mode = args.mode, is_eval = args.eval)
agent = Agent(mode = args.mode, load_file=args.load, is_eval=args.eval)
if(args.eval):
    evaluate(environment, agent, args.mode)
else:
    train(environment, agent, args.mode)