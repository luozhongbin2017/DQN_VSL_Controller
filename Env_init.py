# Environment Construction
import numpy as np
from lib import gym
from lib.sumo.tools import traci
import os,sys
import xml.etree.ElementTree as ET

from lib.gym import error, spaces
from lib.gym import utils
from lib.gym.utils import seeding
from collections import deque
from sumolib import checkBinary

#Environment Constants       
WARM_UP_TIME = 300 * 1e3,
END_TIME = 7500 * 1e3
speeds = [11.11, 13.88, 16.67, 19.44, 22.22]  # possible actions collection

VEHICLE_MEAN_LENGTH = 5

class SumoEnv(gym.Env):       ###It needs to be modified
    def __init__(self, frameskip = 3, evaluation = False):
        #create environment

        self.state_shape = state_shape
        self.action_size = action_size
        self.actions = actions
        self.warmstart = WARM_UP_TIME
        self.warmend = END_TIME
        self.evaluation = evaluation

        self.run_step = 0
        self.lane_list = list()
        self.vehicle_list = list()
        self.vehicle_position = list()
        self.lanearea_dec_list = list()
        self.lanearea_max_speed = dict()
        self.action_set = dict()
        self.pre_reward = 0.0

        # initialize sumo path
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'],'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoBinary = "sumo"
        self.projectFile = './SimulationProject/Scenario_4_v3_VSLdueling/'    

        # initialize lane_list
        net_tree = ET.parse("project/ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])

        # initialize lanearea_dec_list
        dec_tree = ET.parse("project/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
 
         

        # initalize action set
        i = 0
        for lanearea in self.lanearea_dec_list:
            for speed in speeds:
                self.action_set[i] = [lanearea,speed]
                i += 1
        self.action_space = spaces.Discrete(len(self.action_set))

        # initialize vehicle_list and vehicle_position
        run_step = 0
        while run_step< END_TIME + 2:
            self.vehicle_list.append(dict())
            self.vehicle_position.append(dict())
            for lane in net_tree.iter("lane"):
                self.vehicle_list[run_step][lane.attrib["id"]]=list()
                self.vehicle_position[run_step][lane.attrib["id"]]=[0]*int(float(lane.attrib["length"])/VEHICLE_MEAN_LENGTH + 1)
            run_step += 1
    
    def is_episode(self):
        if self.run_step == END_TIME:
            traci.close()
            return True
        for lanearea_dec in self.lanearea_dec_list:
            dec_length=traci.lanearea.getLength(lanearea_dec)
            jam_length=traci.lanearea.getJamLengthMeters(lanearea_dec)
            if jam_length * 0.8 < dec_length:
                traci.close()
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
                vehicle_pos= traci.vehicle.getPostion(vehicle)
                index = (vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH
                self.vehicle_position[self.run_step][lane][index]+=1
        return [self.lane_list, self.vehicle_position]

    def update_observation(self):
        # Update observation of environment state.

        self.update_target_vehicle_set()
        self.transform_vehicle_position()

        current_step_vehicle = list()
        for lane in self.lane_list:
            current_step_vehicle += self.vehicle_list[self.run_step][lane]

        vehicle_speed = dict()
        vehicle_acceleration = dict()
        for vehicle in current_step_vehicle:
            vehicle_speed[vehicle] = traci.vehicle.getSpeed(vehicle)
            vehicle_acceleration[vehicle] = traci.vehicle.getAcceleration(vehicle)
        state = (self.vehicle_position[self.run_step], vehicle_speed, vehicle_acceleration)
        return state
    

    def step_reward(self):

        queue_len = [0] * len(self.lane_list)
        i = 0
        for lane in self.lane_list:
            j = len(self.vehicle_position[self.run_step][lane])
            while True:
                if self.vehicle_position[self.run_step][lane][j]==1:
                    queue_len[i]+=1
                else:
                    break
            i+=1

        i = 0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i += 1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U = queue_len_sum + vehicle_sum
        
        return -(1 * U/3600 - self.pre_reward)
    
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
        observation = self.update_observation()
        reward = self.step_reward()

        self.run_step += 1
        traci.simulationStep()
        return reward, observation, self.is_episode()

    def reset(self):
        # Reset simulation with the random seed randomly selected the pool.
        if self.evaluation == False:
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='evaluation')
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