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
from utils import common

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
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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
    params = common.Constants
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env_traino = Environment.Env()  ###This IO needs to be modified
    state_shape = env.state_shape
    action_size = env.action_size
    env = ptan.common.wrappers.wrap_dqn(env_traino, stack_frames = 3)

    writer = SummaryWriter(comment="-Variable-Speed-Controller-Dueling")
    net = CreateNetwork(state_shape, action_size).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start']) # = def choose actions
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])  #batch sampling
    optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
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
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['max_tau'] == 0:
                tgt_net.sync()  #Sync q_eval and q_target
            
            if frame_idx % 5000 == 0: #Start evaluation
                evaluate_agent(params, tgt_net)
        envo.is_done()
    

if __name__ == '__main__':
    DQNAgent()


#Play and evaluation
def evaluate_agent(params, tgt_net):   ###This needs to be modified        
    tmp_reward = []
    tmp_mainline_time_mean = []
    tmp_ramp_time_mean = []
    tmp_system_time_mean = []          
    for idx_eval in range(3):
        
        env_evalo = Environment.Env(evaluation = True)  ###This IO needs to be modified in file /lib/Environment
        env_eval = ptan.common.wrappers.wrap_dqn(env_evalo)
        reward_sum, mainline_time, ramp_time, _ = evaluate_agent(env_evaluation, scenario_evaluation, tgt_net)
        tmp_reward.append(reward_sum)
        tmp_mainline_time_mean.append(np.mean(mainline_time))
        tmp_ramp_time_mean.append(np.mean(ramp_time))
        tmp_system_time_mean.append(np.mean(mainline_time+ramp_time))
        env_evalo.is_done()

    