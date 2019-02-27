# Environment Construction
import numpy as np
import os,sys
sys.path.append("./lib")
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
END_TIME = 100 * 1e2
VEHICLE_MEAN_LENGTH = 5
speeds = [11.11, 13.89, 16.67, 19.44, 22.22]  # possible actions collection



class SumoEnv(gym.Env):       ###It needs to be modified
    def __init__(self, writer, frameskip= 10, death_factor= 0.001, demonstration = False):
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
        self.lanearea_ob = list()
        self.lane_length = list()
        self.action_set = dict()
        #self.waiting_time = 0.0
        self.delaytime = 0.0
        self.saturation = 0.0
        self.death_factor = death_factor
        #self.ratio = 0.0
        self.meanspeed = 22.22

        # initialize sumo path
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'],'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoBinary = " "
        self.projectFile = './project/'    

        # initialize lane_list and edge_list
        net_tree = ET.parse("./project/ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])
            self.lane_length.append(float(lane.attrib["length"]))
        #self.state_shape = (3, len(self.lane_list), 441)
        self.observation_space = spaces.Box(low= -1, high=100, shape=(3 * len(self.lane_list), 441, 1), dtype=np.float32)

        # initialize lanearea_dec_list
        dec_tree = ET.parse("./project/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            if lanearea_dec.attrib["freq"] == '60':
                self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
                self.lanearea_max_speed[lanearea_dec.attrib["id"]] = 22.22
            else:
                self.lanearea_ob.append(lanearea_dec.attrib["id"])
 
         

        # initalize action set
        i = 0
        j = 0
        lanearea = list()
        while i < len(self.lanearea_dec_list):
            lanearea.append(self.lanearea_dec_list[i])
            if (i + 1) % 3 == 0:
                for speed in speeds:
                    self.action_set[j] = [lanearea.copy(), speed]
                    j += 1
                lanearea.clear()
            i += 1
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

    def is_episode(self):
        if self.run_step == END_TIME:
            print('Scenario ends... at phase %d' % (self.run_step / 1800 + 1))
            traci.close(False)
            return True
        if self.run_step % 1800 == 0:
            self.death_factor -= self.death_factor / 5
        if self.death_factor < self.saturation:
            print('You are jammed to Death! Scenario ends at phase %d' % (self.run_step / 1800 + 1))
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
    
    def _delaytime(self):
        ms = list()
        for lane in self.lane_list:
            ms.append(traci.lane.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        targetspeed = 22.22
        delaytime = np.sum(self.lane_length) * (1 / meanspeed - 1/targetspeed)
        return delaytime
    
    def _getsaturation(self):
        saturation = list()
        for lane in self.lane_list:
            saturation.append(traci.lane.getLastStepVehicleNumber(lane)/(traci.lane.getLength(lane) / 5))
        ans = np.mean(saturation)
        print("saturation:" + str(ans))
        self.saturation = ans

    def _transformedtanh(self, x):
        return (np.exp(-x/4) - np.exp(x/4))/(np.exp(x/4) + np.exp(-x/4))
    
    def step_reward(self):
        #Using waiting_time to present reward.
        delta_delay = self._delaytime() - self.delaytime
        delay_reward = self._transformedtanh(delta_delay)
        satur_discount = 1 + np.sign(delta_delay) * (self.death_factor - self.saturation) ** 2
        self.delaytime = self._delaytime()
        if self.saturation > 0.75:
            reward = -1.0
        else:
            reward = 0.1 * delay_reward * satur_discount * 0.06
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
        for i in range(3):
            self.lanearea_max_speed[action[0][i]]=action[1]
        self.reset_vehicle_maxspeed()
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.step_reward()
            traci.simulationStep()
            num_arrow = int(self.run_step * 50 / END_TIME)
            num_line = 50 - num_arrow
            percent = self.run_step * 100.0 / END_TIME
            process_bar = 'Scenario Running... [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
            sys.stdout.write(process_bar)
            sys.stdout.flush()
            self.run_step += 1
        observation = self.update_observation()
        #print("reward:" + str(np.mean(np.clip(reward, -1, 1))))
        return observation, reward, self.is_episode(), {'No info'}

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
            self.scenario = traci.getConnection('demonstration')

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