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
speeds = [11.11, 13.88, 16.67, 19.44, 22.22]

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
        self.lanearea_dec_list=list()
        self.lanearea_max_speed=dict()
        self.action_set=list()

        # initialize sumo path
        
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'],'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        self.sumoBinary = "sumo"
        self.projectFile = './SimulationProject/Scenario_4_v3_VSLdueling/'    

        # initialize lane_list
        net_tree = ET.parse("ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])

        # initialize lanearea_dec_list
        dec_tree = ET.parse("ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
 
         

        # initalize action set
        for lanearea in self.lanearea_dec_list:
            for speed in speeds:
                self.action_set.append([lanearea,speed])

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
            traci.start([self.sumoBinary, '-c', self.projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            np.random.seed(42)
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size=N_SIM_EVAL)
            traci.start([self.sumoBinary, '-c', self.projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('evaluation')
        self.run_step = 0
    
    def new_episode(self,):
        pass
    
    def is_episode(self):
        for lanearea_dec in self.lanearea_dec_list:
            dec_length=traci.lanearea.getLength(lanearea_dec)
            jam_length=traci.lanearea.getJamLengthMeters(lanearea_dec)
            if jam_length * 0.8 > dec_length:
                return True
        
        if self.run_step == END_TIME:
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
        return 
    


    def update_observation(self):
        # Update observation of environment state.
        pass
        # Your codes are here.
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
        
        return self.vehicle_position[self.run_step],vehicle_speed,vehicle_acceleration
    

    def step_reward(self):
        threshold = 101 #it needs to be modified
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

        U = queue_len_sum + vehicle_sum

        min_speed = min(lane_speed)

        if min_speed > threshold:
            return 0
        else:
            return -1*U/3600

        pass

    
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

        pass


    def step(self, a):
        # Conduct action, update observation and collect reward.
        action = self.action_set[a]
        self.lanearea_max_speed[action[0]]=action[1]
        self.reset_vehicle_maxspeed()
        observation = self.update_observation()
        reward = self.step_reward()

        self.run_step += 1
        traci.simulationStep()
        return reward,observation,self.is_episode()



    def reset(self,evaluation):
        # Start simulation with the random seed randomly selected the pool.
        if evaluation == False:
            np.random.seed(43)
            N_SIM_TRAINING = 20
            random_seeds_training = np.random.randint(low=0, high=1e5, size=N_SIM_TRAINING)
            traci.start([self.sumoBinary, '-c', self.projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            np.random.seed(42)
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size=N_SIM_EVAL)
            traci.start([self.sumoBinary, '-c', self.projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('evaluation')

        self.warm_up_simulation()

        self.run_step = 0

        return self.update_observation()



    def render(self):
        pass

    def frame_buffer(self,):
        pass
    
    def is_done(self,):
        traci.close()
        env_closer.close()
        return

#Image preprocessing (should i keep this stage?) -> dataset(ndarray)
    
