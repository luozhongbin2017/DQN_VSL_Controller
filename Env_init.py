# Environment Construction
import numpy as np
from lib.sumo.tools import traci
import os,sys
import xml.etree.ElementTree as ET
from collections import deque
from sumolib import checkBinary
from common import closer

env_closer = closer.Closer()

#Environment Constants       
state_shape = (3,20,880)             # Our input is a stack of 3 frames hence 20x880x3 (Width, height, channels)
WARM_UP_TIME = 300 * 1e3,
END_TIME = 7500 * 1e3
action_size = 5                # 5 possible actions
actions = [40, 50, 60, 70, 80] # possible actions collection

VEHICLE_MEAN_LENGTH = 5

class Env():       ###It needs to be modified
    def __init__(self, evaluation=False):
        #create environment

        self.state_shape = state_shape
        self.action_size = action_size
        self.actions = actions
        self.warmstart = WARM_UP_TIME
        self.warmend = END_TIME

        self.run_step=0
        self.lane_list=list()
        self.vehicle_list=list()
        self.vehicle_position=list()
        

        # initialize sumo path
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'],'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        sumoBinary = "sumo"
        projectFile = './SimulationProject/Scenario_4_v3_VSLdueling/'    

        # initialize lane_list
        net_tree = ET.parse("ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])

        # initialize vehicle_list and vehicle_position
        run_step = 0
        while run_step< END_TIME + 2:
            self.vehicle_list.append(dict())
            self.vehicle_position.append(dict())
            for lane in net_tree.iter("lane"):
                self.vehicle_list[run_step][lane.attrib["id"]]=list()
                self.vehicle_position[run_step][lane.attrib["id"]]=[0]*int(float(lane.attrib["length"])/VEHICLE_MEAN_LENGTH + 1)
            run_step += 1
        
        # Start simulation with the random seed randomly selected the pool.
        if evaluation == False:
            np.random.seed(43)
            N_SIM_TRAINING = 20
            random_seeds_training = np.random.randint(low=0, high=1e5, size=N_SIM_TRAINING)
            traci.start([sumoBinary, '-c', projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            np.random.seed(42)
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size=N_SIM_EVAL)
            traci.start([sumoBinary, '-c', projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('evaluation')
        self.run_step = 0
    
    def new_episode(self,):
        pass
    
    def warm_up_simulation(self, traci):
        # Warm up simulation.
        warm_step=0
        while warm_step < WARM_UP_TIME:
            traci.simulationStep()
            warm_step += 1
    
    def update_target_vehicle_set(self, traci):
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
                self.vehicle_position[self.run_step][lane]+=1
        return 
    

    def update_observation(self, traci):
        # Update observation of environment state.
        pass
        # Your codes are here.
        pass
        return 0
    

    def step_reward(self):
        lane_speed=[0]*len(self.lane_list)
        i = 0

        for lane in self.lane_list:
            cur_speed_sum = 0
            for vehicle in self.vehicle_list[self.run_step][lane]:
                cur_speed_sum += traci.vehicle.getSpeed(vehicle)

            lane_speed[i]=cur_speed_sum / len(self.vehicle_list[self.run_step][lane])
            i += 1

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

        i=0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i+=1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U=queue_len_sum+vehicle_sum

        min_speed=min(lane_speed)
        if min>U:
            return 0
        else:
            return -1*U/3600

        pass

    
    def step(self, traci, action):
        # Conduct action, update observation and collect reward.

        # change lane speed
        for (lane,maxSpeed) in action:
            traci.lane.setMaxSpeed(lane,maxSpeed)

        reward=0
        # reward is unkonwn
        observation=self.update_observation(traci)
        return reward,observation

    def frame_buffer(self,):
        pass
    
    def is_done(self,):
        traci.close()
        pass

#Image preprocessing (should i keep this stage?) -> dataset(ndarray)
    
