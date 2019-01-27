# Environment Construction
import numpy as np
from lib.sumo.tools import traci
import os,sys
import xml.etree.ElementTree as ET
from collections import deque
from sumolib import checkBinary
from common import closer
from lib import error, logger, seeding

env_closer = closer.Closer()

#Environment Scalar      
state_shape = (3,20,880)             # Our input is a stack of 3 frames hence 20x880x3 (Width, height, channels)
WARM_UP_TIME = 300 * 1e3,
END_TIME = 7500 * 1e3
action_size = 5                # 5 possible actions
action_set = np.zeros(1, dtype = int) + (60, 70, 80, 90, 100) # possible actions collection (km/h)

VEHICLE_MEAN_LENGTH = 5
env_closer = closer.Closer()

# Env-related abstractions

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        logger.warn("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

# Space-related abstractions

class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    def __init__(self, shape=None, dtype=None):
        import numpy as np # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    __contains__ = contains

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n

warn_once = True

def deprecated_warn_once(text):
    global warn_once
    if not warn_once: return
    warn_once = False
    logger.warn(text)

class Wrapper(Env):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        if hasattr(self, "_step"):
            deprecated_warn_once("%s doesn't implement 'step' method, but it implements deprecated '_step' method." % type(self))
            self.step = self._step
            return self.step(action)
        else:
            deprecated_warn_once("%s doesn't implement 'step' method, " % type(self) +
                "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
            return self.env.step(action)

    def reset(self, **kwargs):
        if hasattr(self, "_reset"):
            deprecated_warn_once("%s doesn't implement 'reset' method, but it implements deprecated '_reset' method." % type(self))
            self.reset = self._reset
            return self._reset(**kwargs)
        else:
            deprecated_warn_once("%s doesn't implement 'reset' method, " % type(self) +
                "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
            return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        if self.env:
            return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec

class ObservationWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        deprecated_warn_once("%s doesn't implement 'observation' method. Maybe it implements deprecated '_observation' method." % type(self))
        return self._observation(observation)

class RewardWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        deprecated_warn_once("%s doesn't implement 'reward' method. Maybe it implements deprecated '_reward' method." % type(self))
        return self._reward(reward)

class ActionWrapper(Wrapper):
    def step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def action(self, action):
        deprecated_warn_once("%s doesn't implement 'action' method. Maybe it implements deprecated '_action' method." % type(self))
        return self._action(action)

    def reverse_action(self, action):
        deprecated_warn_once("%s doesn't implement 'reverse_action' method. Maybe it implements deprecated '_reverse_action' method." % type(self))
        return self._reverse_action(action)

class Discrete(Space):
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        return gym.spaces.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n

class SumoEnv(Env):       ###It needs to be modified
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, evaluation=False, obs_type = 'list', framestack = 3, repeat_action_probability=0., full_action_space=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""
        #create environment

        self.action_set = action_set
        self.action_space = Discrete(len(action_set))

        self._obs_type = obs_type
        self.frameskip = framestack
        self.observation_space = 0 ###it needs to be modified after disscussion

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
        net_tree = ET.parse("./project/ramp.net.xml")
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
            np.random.seed(self.seed()[0])
            N_SIM_TRAINING = 20
            random_seeds_training = np.random.randint(low=0, high=1e5, size=N_SIM_TRAINING)
            traci.start([sumoBinary, '-c', projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            np.random.seed(self.seed()[1])
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size=N_SIM_EVAL)
            traci.start([sumoBinary, '-c', projectFile+'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('evaluation')
        self.run_step = 0
    
    def seed(self, seed = None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]
    
    def new_episode(self,):
        pass
    
    def reset(self, traci):
        # Warm up simulation.
        warm_step=0
        while warm_step < WARM_UP_TIME:
            traci.simulationStep()
            warm_step += 1
        self.update_observation()
    
    def update_target_vehicle_set(self, traci):
        # Update vehicle ids in the target area
        for lane in self.lane_list:
            self.vehicle_list[self.run_step][lane] = traci.lane.getLastStepVehicleIDs(lane)
    
    def transform_vehicle_position(self, traci):
        # Store vehicle positions in matrices.
        for lane in self.lane_list:
            lane_shape = traci.lane.getShape(lane)
            for vehicle in self.vehicle_list[self.run_step][lane]:
                vehicle_pos= traci.vehicle.getPostion(vehicle)
                index = (vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH
                self.vehicle_position[self.run_step][lane]+=1
        return 

    @property
    def _n_actions(self):
        return len(self._action_set)

    def update_observation(self, traci):
        # Update observation of environment state.
        pass
        # Your codes are here.
        pass
        return 0
    

    def act(self, traci, action):
        #modified suggestions: add a step forward!

        # change lane speed
        for (lane,maxSpeed) in action:
            traci.lane.setMaxSpeed(lane, maxSpeed/3.6)

        threshold = 80/3.6 #it needs to be modified
        lane_speed = [0]*len(self.lane_list)
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

    
    def step(self, traci, action):
        # Collect observation and collect reward.
        # First we do some framestack
        num_steps = self.frameskip
        for _ in range(num_steps):
            reward += self.act(traci, action)
        observation = self.update_observation(traci)
        return observation, reward, self.is_done(), {"mean_vehicle_speed": self.vehicle_position} #needs to be modified!!!

    def is_done(self,):
        #This function helps to identify whether a sumo episode is done.
        pass

    def frame_buffer(self,):
        pass
    
    def close(self,):
        traci.close()
        env_closer.close()
        return
    
