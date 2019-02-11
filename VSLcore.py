# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:36:09 2019

@author: ChocolateDave
"""

#Import Modules
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os,sys
sys.path.append("lib")
sys.path.append("common")

import Env_init as Environment
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from common import action, agent, utils, experience, tracker, wrapper

#Global Variable:
params = utils.Constants

#Build Up Dueling Neural Network
class DuelingNetwork(nn.Module):
    """
    Create a neural network to convert image data
    """
    def __init__(self, input_shape, n_actions):
        super(DuelingNetwork, self).__init__()

        self.Convolutional_Layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        #print('input_shape[0]: ', input_shape[0])
        conv_out_size = self._get_conv_out(input_shape)
        self.Fully_Connected_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.Fully_Connected_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        #print('1, *shape: ', torch.zeros(1, *shape).size())
        o = self.Convolutional_Layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
        #print(x.size())
        conv_out = self.Convolutional_Layer(fx).view(fx.size()[0], -1)
        val = self.Fully_Connected_val(conv_out)
        adv = self.Fully_Connected_adv(conv_out)
        return val + adv - adv.mean()

#Training
def DQNAgent():
    print("Cuda's availability is %s" % torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Environment.SumoEnv(device)  ###This IO needs to be modified
    env = env.unwrapped
    #print(env_traino.state_shape)
    #env = wrapper.wrap_dqn(env_traino, stack_frames = 3)  ###wrapper needs to be modified

    writer = SummaryWriter(comment = '-VSL-Dueling-')
    net = DuelingNetwork(env.state_shape, env.action_space.n).to(device)
    env_graph = Environment.SumoEnv(device)
    env_graph = env_graph.unwrapped
    writer.add_graph(DuelingNetwork(env_graph.state_shape, env_graph.action_space.n), (env_graph.reset(),))
    if env_graph.run_step > 0:
        env_graph.close()
    tgt_net = agent.TargetNet(net)
    selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = tracker.EpsilonTracker(selector, params)
    agents = agent.DQNAgent(net, selector, device = device)

    exp_source = experience.ExperienceSourceFirstLast(env, agents, gamma=params['gamma'], steps_count=1)
    buffer = experience.PrioritizedReplayBuffer(exp_source, buffer_size=params['replay_size'], alpha = 0.6)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with tracker.RewardTracker(writer, params['stop_reward'], params['stop_frame']) as reward_tracker:  #stop reward needs to be modified according to reward function
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    env.close()
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'], beta = 0.4)
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            #Writer function -> Tensorboard file
            writer.add_scalar("Loss", loss_v, frame_idx)

            #Demonstrate function -> Visualization
            '''if frame_idx % 5000 == 0:  #start evaluation
                for i in range(3): 
                    evaluation_agent(device, net)'''
            
            #saving model
            if frame_idx % 10000 == 0:
                pass
            
            if frame_idx % params['max_tau'] == 0:
                tgt_net.sync()  #Sync q_eval and q_target
    

if __name__ == '__main__':
    DQNAgent()

def evaluation_agent(device, neural_network):
    env_evalo = Environment.SumoEnv(evaluation = True)
    eval_writer = SummaryWriter(log_dir = './logs/evaluation', comment = '-Variable-Speed-Controller-Dueling')
    eval_selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    eval_epsilon_tracker = tracker.EpsilonTracker(eval_selector, params)
    eval_agent = agent.DQNAgent(neural_network, eval_selector, device = device)
    exp_source = experience.ExperienceSourceFirstLast(env_evalo, eval_agent, gamma=params['gamma'], steps_count=1)
    buffer = experience.PrioritizedReplayBuffer(exp_source, buffer_size=params['replay_size'], alpha = 0.6)

    eval_idx = 0

    with tracker.RewardTracker(eval_writer, params['stop_reward'], params['stop_frame']) as reward_tracker:
        while True:
            eval_idx += 1
            buffer.populate(1)
            eval_epsilon_tracker.frame(eval_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], eval_idx, eval_selector.epsilon):
                    break

