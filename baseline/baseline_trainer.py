import torch
import gym
import random
import numpy as np
import datetime
import torch.multiprocessing as mp
import cv2

from collections import deque
from baseline.utils import Json_Config_Parser

from torch.utils.tensorboard import SummaryWriter

class Policy:

    def __init__(self, file_name):
        self.parser = Json_Config_Parser(file_name)
        self.parm = self.parser.load_parser()
        self.agent_parms = self.parser.load_agent_parser()
        self.optimizer_parms = self.parser.load_optimizer_parser()

        self.env = gym.make(self.parm['env_name'])

        self.obs_set = deque(maxlen=self.parm['state_size'][0])
        self.replay_memory = deque(maxlen=int(self.parm['size_replay_memory']))
        self.discount_factor = self.parm['discount_factor']


        # -------------------------------------------------------------------------#
        if 'linear_decay' in self.optimizer_parms.keys():
            self.decaying_mode = self.optimizer_parms['linear_decay'] == 'True'
        else:
            self.decaying_mode = False
        if 'learning_rate' in self.optimizer_parms.keys():
            self.lr = self.optimizer_parms['learning_rate']
        if 'Exploration_method' in self.parm.keys():
            self.Exploration_method = self.parm['Exploration_method']
        if 'epsilon_min' in self.parm.keys():
            self.epsilon_min = self.parm['epsilon_min']
        # -------------------------------------------------------------------------#

        if 'conservative_mode' in self.parm.keys():
            self.c_mode = self.parm['conservative_mode'] == 'True'
            self.tau = self.parm['tau']
        else:
            self.c_mode = False
            self.update_step = self.parm['update_step']

        # -------------------------------------------------------------------------#


        self.state_size = self.parm['state_size']
        self.action_size = self.parm['action_size']

        self.gpu = torch.device(self.parm['gpu_name'])
        self.cpu = torch.device('cpu')
        self.batch_size = self.parm['batch_size']

        self.run_step = self.parm['run_step']
        self.start_step = self.parm['start_step']
        self.show_episode = self.parm['show_episode']

        self.save_path = self.parm['save_path']
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.save_path = self.save_path + self.parm['env_name'] + '_' + date_time + '.pth'
        if 'reward_scaling' in self.parm.keys():
            self.reward_scaling = self.parm['reward_scaling']

        name = self.parm['tensorboard_path'] + self.parm['env_name'] + '_' + date_time + '_' + 'reward_scaling_' + str(
            self.reward_scaling)
        if 'fixed_temperature' in self.parm.keys():
            if self.parm['fixed_temperature'] == 'True':
                name += '_Fixed'
        self.writer = SummaryWriter(name + '/')

        self.inference_mode = self.parm['inference_mode'] == 'True'

        if len(self.state_size) >= 3:
            self.mode = 'Image'
        else:
            self.mode = 'Vector'

    def reset(self):
        for i in range(self.state_size[0]):
            self.obs_set.append(np.zeros([i for i in self.state_size[1:]]))

    def append_memory(self,sarsd):
        self.replay_memory.append(sarsd)

    def generate_optimizer(self):
        pass

    def lr_scheduler(self, step):
        # p = step / self.run_step / 10
        # self.lr = self.optimizer_parms['learning_rate'] * (1 - p)
        #
        # for g in self.optimizer.param_groups:
        #     g['lr'] = self.lr

        pass

    def preprocess_state(self, obs):

        state = np.zeros(self.parm['state_size'])

        if self.mode == 'Image':
            obs = cv2.resize(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
        self.obs_set.append(obs)

        for i in range(self.parm['state_size'][0]):
            state[i] = self.obs_set[i]

        if self.mode == 'Image':
            state = np.uint8(state)
        return state

    def get_action(self, state):

        pass

    def target_network_update(self, step):
        # if self.c_mode:
        #     with torch.no_grad():
        #         for t_pa, pa in zip(self.target_agent.parameters(), self.agent.parameters()):
        #             temp = self.tau * pa + (1 - self.tau) * t_pa
        #             t_pa.copy_(temp)
        # elif step % self.update_step == 0:
        #     self.target_agent.load_state_dict(self.agent.state_dict())
        pass

    def train(self):
        pass

    def run(self):
        pass




