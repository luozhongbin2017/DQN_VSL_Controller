import sys
import time
import numpy as np
import torch
import torch.nn as nn

class RewardTracker:
    def __init__(self, writer, stop_reward, stop_frame):
        self.writer = writer
        self.stop_reward = stop_reward
        self.stop_frame = stop_frame

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: Just done %d episode(s), mean reward %d, speed %.2f f/s%s" % (
            frame, len(self.total_rewards),mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("Interaction/epsilon", epsilon, frame)
        self.writer.add_scalar("Interaction/speed", speed, frame)
        self.writer.add_scalar("Interaction/reward_100", mean_reward, frame)
        self.writer.add_scalar("Interaction/reward", reward, frame)
        if mean_reward > self.stop_reward or  frame > self.stop_frame:
            print("Training finished in %d steps!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)