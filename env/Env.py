# Environment Construction
import numpy as np
import os,sys
import xml.etree.ElementTree as ET

import cmath
import gym
import traci
import torch
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from collections import deque
from sumolib import checkBinary

#Environment Constants
STATE_SHAPE = (81, 441, 1)      
WARM_UP_TIME = 3 * 1e2
END_TIME = 105 * 1e2
VEHICLE_MEAN_LENGTH = 5
speeds = [11.11, 13.88, 16.67, 19.44, 22.22]  # possible actions collection



class SumoEnv(gym.Env):       ###It needs to be modified
    def __init__(self, writer, frameskip = 3, demonstration = False):
        #create environment

        self.warmstart = WARM_UP_TIME
        self.warmend = END_TIME
        self.demonstration = demonstration
        self.frameskip = frameskip
        self.writer = writer

        self.run_step = 0
        self.lane_list = list()
        self.edge_list = list()
        self.vehicle_list = list()
        self.vehicle_position = list()
        self.lanearea_dec_list = list()
        self.lanearea_max_speed = dict()
        self.action_set = dict()
        self.waiting_time = 0.0
        self.death_factor = 0.001
        self.ratio = 0.0

        # initialize sumo path
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'],'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoBinary = " "
        self.projectFile = './project/'    

        # initialize lane_list and edge_list
        net_tree = ET.parse("/env/ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])
        #self.state_shape = (3, len(self.lane_list), 441)
        self.observation_space = spaces.Box(low= -1, high=100, shape=(3 * len(self.lane_list), 441, 1), dtype=np.float32)

        # initialize lanearea_dec_list
        dec_tree = ET.parse("/env/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
 
         

        # initalize action set
        i = 0
        for lanearea in self.lanearea_dec_list:
            for speed in speeds:
                self.action_set[i] = [lanearea,speed]
                i += 1
        #print(self.action_set)
        self.action_space = spaces.Discrete(len(self.action_set))

        # initialize vehicle_list and vehicle_position
        run_step = 0
        while run_step< END_TIME + 2:
            self.vehicle_list.append(dict())
            self.vehicle_position.append(dict())
            for lane in net_tree.iter("lane"):
                self.vehicle_list[run_step][lane.attrib["id"]]=list()
                
                self.vehicle_position[run_step][lane.attrib["id"]]=[0]*int(float(lane.attrib["length"])/VEHICLE_MEAN_LENGTH + 2)
            run_step += 1
    
    def getstatus(self):
        self.writer.add_scalar('Waiting time', self.waiting_time)
        self.writer.add_scalar('Congestion ratio', self.ratio)
    
    def is_episode(self):
        if self.run_step == END_TIME:
            print('You survived! Simulation end at phase %d' % (self.run_step / 1800 + 1))
            self.getstatus()
            traci.close(False)
            return True
        if self.run_step % 1800 == 0:
            self.death_factor -= 0.0002
            print('phase: %d' % (self.run_step / 1800 + 1), ' step: %d' % self.run_step)
        for lanearea_dec in self.lanearea_dec_list:
            dec_length = 0
            jam_length = 0
            dec_length += traci.lanearea.getLength(lanearea_dec)
            jam_length += traci.lanearea.getJamLengthMeters(lanearea_dec)
        self.ratio = jam_length / dec_length
        if self.death_factor < self.ratio:
            print('You are jammed to Death! Game finished at phase %d' % (self.run_step / 1800 + 1))
            self.getstatus()
            traci.close(False)
            return True
        return False

    def warm_up_simulation(self):
        # Warm up simulation.
        warm_step=0
        while warm_step < WARM_UP_TIME:
            traci.simulationStep()
            warm_step += 1
    
    def update_target_vehicle_set(self):
        # Update vehicle ids in the target area
        for lane in self.lane_list:
            self.vehicle_list[self.run_step][lane] = traci.lane.getLastStepVehicleIDs(lane)
    
    def transform_vehicle_position(self):
        # Store vehicle positions in matrices.
        for lane in self.lane_list:
            lane_shape = traci.lane.getShape(lane)
            for vehicle in self.vehicle_list[self.run_step][lane]:
                vehicle_pos= traci.vehicle.getPosition(vehicle)
                index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))
                self.vehicle_position[self.run_step][lane][index]=1
        return [self.lane_list, self.vehicle_position]

    def update_observation(self):
        # Update observation of environment state.

        self.update_target_vehicle_set()
        self.transform_vehicle_position()

        lane_map = [0] * len(self.lane_list)
        i=0
        for lane in self.lane_list:
            lane_map[i]=lane
            i+=1

        vehicle_position = np.zeros((len(self.lane_list),441),dtype = np.float32)
        vehicle_speed = np.zeros((len(self.lane_list),441),dtype = np.float32)
        vehicle_acceleration = np.zeros((len(self.lane_list),441),dtype = np.float32)
        state = np.empty((1, 3* len(self.lane_list), 441), dtype = np.float32)
        #self.state_shape = torch.from_numpy(state).shape if device == "cuda" else state.shape
    
        for lane in self.lane_list:
            lane_index = lane_map.index(lane)
            lane_len = traci.lane.getLength(lane)
            lane_stop = int (lane_len / VEHICLE_MEAN_LENGTH)
            for i in range(lane_stop, 440):
                vehicle_position[lane_index][i] = vehicle_speed[lane_index][i] = vehicle_acceleration[lane_index][i] = -1.0

        current_step_vehicle = list()
        for lane in self.lane_list:
            current_step_vehicle += self.vehicle_list[self.run_step][lane]

        for vehicle in current_step_vehicle:
            vehicle_in_lane = traci.vehicle.getLaneID(vehicle)
            lane_index = lane_map.index(vehicle_in_lane)
            vehicle_pos= traci.vehicle.getPosition(vehicle)
            lane_shape = traci.lane.getShape(vehicle_in_lane)
            vehicle_index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))

            vehicle_position[lane_index][vehicle_index] = 1.0
            vehicle_speed[lane_index][vehicle_index] = traci.vehicle.getSpeed(vehicle) 
            vehicle_acceleration[lane_index][vehicle_index] = traci.vehicle.getAcceleration(vehicle)
        state[0] = np.concatenate((vehicle_position, vehicle_speed, vehicle_acceleration), axis= 0)
        state = np.swapaxes(state, 2, 0)
        #print(state.shape)
        return state
    
 
    def step_reward(self):
        #Using waiting_time to present reward.

        wt = list()
        for lane in self.lane_list:
            #print(traci.lane.getWaitingTime(lane))
            wt.append(traci.lane.getWaitingTime(lane))
        self.waiting_time += np.sum(wt)
        reward = -10 if np.sum(wt) != 0 else 1
        #print(reward)
        return reward
    
    def reset_vehicle_maxspeed(self):
        for lane in self.lane_list:
            max_speed = traci.lane.getMaxSpeed(lane)
            for vehicle in self.vehicle_list[self.run_step][lane]:
                traci.vehicle.setMaxSpeed(vehicle,max_speed)
        
        for dec_lane in self.lanearea_dec_list:
            vehicle_list = traci.lanearea.getLastStepVehicleIDs(dec_lane)
            max_speed = self.lanearea_max_speed[dec_lane]
            for vehicle in vehicle_list:
                traci.vehicle.setMaxSpeed(vehicle,max_speed)

    def step(self, a):
        # Conduct action, update observation and collect reward.
        reward = 0.0
        action = self.action_set[a]
        self.lanearea_max_speed[action[0]]=action[1]
        self.reset_vehicle_maxspeed()
        for _ in range(self.frameskip):
            reward += self.step_reward()
            traci.simulationStep()
            self.run_step += 1
        observation = self.update_observation()
        #print(a, reward)
        return observation, reward, self.is_episode(), {"Waiting_time": self.waiting_time}

    def reset(self):
        # Reset simulation with the random seed randomly selected the pool.
        if self.demonstration == False:
            self.sumoBinary = "sumo"
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            self.sumoBinary = "sumo-gui"
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='demonstration')
            self.scenario = traci.getConnection('evaluation')

        self.warm_up_simulation()

        self.run_step = 0

        return self.update_observation()
    
    def seed(self, seed= None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def render(self):
        pass
    
    def closer(self):
        #Meant for forced close ops.
        traci.close(False)