# DQN_VSL_Controller
This is a research project aiming at introducing Deep-Q-Learning methonds to help improve variable speed limitation management in traffic control system.

Log Dec 17th, 2018:
Objects to finish including filling up unfinished structure and building up connections between traci and the structure.

Log Jan 7th, 2019:
The framework has been constructed. However, the evaluation system needs to be fill in and the connection shall be confirmed working.

Log Jan 28th, 2019:
This structure has referenced to gym-based environment. Thus, the construction and wrapping methods must follow an instruction foward by the credits of gym.

Log Feb 9th, 2019:
1. Reward function modified.
previous:
    def step_reward(self):

        threshold = 101                   #it needs to be modified
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

current:
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

        i=0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i+=1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U = queue_len_sum + vehicle_sum
        
        return -(1 * U/3600 - self.pre_reward)
2. Sumo connection modification needs to be confirmed
original:
        if evaluation == False:
            
            N_SIM_TRAINING = 20
            random_seeds_training = np.random.randint(low=0, high=1e5, size= N_SIM_TRAINING)
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size= N_SIM_EVAL)
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='evaluation')
            self.scenario = traci.getConnection('evaluation')

modified:(correct?)
    def reset(self, evaluation = self.evaluation):
        # Reset simulation with the random seed randomly selected the pool.
        if evaluation == False:
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

3. Model saving function needs to be appended.

Log Feb 11th, 2019
1. step_reward modified.
previous:
def step_reward(self):

        queue_len = [0] * len(self.lane_list)
        i = 0
        for lane in self.lane_list:
            j = len(self.vehicle_position[self.run_step][lane])
            while True:
                if self.vehicle_position[self.run_step][lane][j] == 1:
                    queue_len[i] += 1
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

modified:
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

Finally, it's all done!
