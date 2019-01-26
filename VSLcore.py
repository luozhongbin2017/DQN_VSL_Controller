# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:36:09 2019

@author: ChocolateDave
"""

#Import Modules
import argparse
import numpy as np

import Env_init as Environment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import ptan
from common import action, agent, utils, experience, tracker, wrapper

#Global Variable:
params = utils.Constants

#Build Up Dueling Neural Network
class CreateNetwork(nn.Module):
    """
    Create a neural network to convert image data
    """
    def __init__ (self, input_shape, n_actions):
        super(CreateNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.Tensor.new_zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean() #action_value

#Training
def DQNAgent():
    print("Cuda's availability is %s" % torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_traino = Environment.Env()  ###This IO needs to be modified
    state_shape = env_traino.state_shape
    action_size = env_traino.action_size
    #env = wrapper.wrap_dqn(env_traino, stack_frames = 3)  ###wrapper needs to be modified

    writer = SummaryWriter(log_dir = './logs/training', comment = '-Variable-Speed-Controller-Dueling')
    net = CreateNetwork(state_shape, action_size).to(device)
    tgt_net = agent.TargetNet(net)
    selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = tracker.EpsilonTracker(selector, params)
    agent = agent.DQNAgent(net, selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env_traino, agent, gamma=params['gamma'], steps_count=1)
    buffer = experience.PrioritizedReplayBuffer(exp_source, buffer_size=params['replay_size'], alpha = 0.6)
    optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with tracker.RewardTracker(writer, params['stop_reward']) as reward_tracker:  #stop reward needs to be modified according to reward function
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'], beta = 0.4)
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            #Writer function -> Tensorboard file
            writer.add_scalars('Training', {'Loss': loss_v, 'Total Reward': new_rewards[0]}, global_step = frame_idx)

            #Evaluation function -> Tensorboard file
            if frame_idx % 5000 == 0:  #start evaluation
                for i in range(3):
                    evaluation_agent(device, net)
                


            if frame_idx % params['max_tau'] == 0:
                tgt_net.sync()  #Sync q_eval and q_target

        env_traino.is_done()
    

if __name__ == '__main__':
    DQNAgent()

def evaluation_agent(device, neural_network):
    env_evalo = Environment.Env(evaluation = True)
    eval_writer = SummaryWriter(log_dir = './logs/evaluation', comment = '-Variable-Speed-Controller-Dueling')
    eval_selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    eval_epsilon_tracker = tracker.EpsilonTracker(eval_selector, params)
    eval_agent = agent.DQNAgent(neural_network, eval_selector, device = device)
    exp_source = experience.ExperienceSourceFirstLast(env_evalo, agent, gamma=params['gamma'], steps_count=1)
    buffer = experience.PrioritizedReplayBuffer(exp_source, buffer_size=params['replay_size'], alpha = 0.6)

    eval_idx = 0

    with tracker.RewardTracker(eval_writer, params['stop_reward']) as reward_tracker:
        while True:
            eval_idx += 1
            buffer.populate(1)
            eval_epsilon_tracker.frame(eval_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], eval_idx, eval_selector.epsilon):
                    break
            
            eval_writer.add_scalars('Evaluation', {'Total_reward': new_rewards}, global_step = eval_idx)

        env_evalo.is_done()

