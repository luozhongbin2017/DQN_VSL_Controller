# Environment Construction
import numpy as np
from lib.sumo.tools import traci
import os,sys
import xml.etree.ElementTree as ET
from collections import deque
from skimage import transform
from sumolib import checkBinary
from utils import closer

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
        net_tree = ET.parse("net.xml")
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
        pass

#Image preprocessing (should i keep this stage?) -> dataset(ndarray)
    
def preprocess_frame(frame):
    
    # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    LEFT, RIGHT = -110, 689
    UP, DOWN = 57, 38
    cropped_frame = frame[UP: DOWN, LEFT: RIGHT]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [20,880])
    
    return preprocessed_frame # 20x880x1 frame

# Stack images for giving have a sense of motion to our Neural Network.
    
stack_size = 3 # We stack 3 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen = 3) 

def stack_frames(stacked_frames, state, is_new_episode):  #Abandoned
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen = 3)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames