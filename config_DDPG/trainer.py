import torch
import random
import numpy as np

from baseline.baseline_trainier import Policy
from config_DDPG.Agent import DDPG_Agent
from baseline.utils import get_optimizer,OUNoise, calculate_global_norm, clipping_by_global_norm
from torchsummary import summary

class action_space:

    def __init__(self, action_size):
        self.shape = [action_size]
        self.low = -1
        self.high = 1

class DDPG_Trainer(Policy):


    def __init__(self, file_name):
        super(DDPG_Trainer, self).__init__(file_name)

        self.agent = DDPG_Agent(self.agent_parms).train()
        self.target_agent = DDPG_Agent(self.agent_parms).eval()


        self.actor, self.critic = self.agent.actor, self.agent.critic
        self.target_actor, self.target_critic = self.target_agent.actor, self.target_agent.critic

        self.target_actor = self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer, self.critic_optimizer = self.generate_optimizer()

        if len(self.parm['state_size']) < 3:
            zz = np.zeros((self.parm['state_size'][0] * self.parm['state_size'][1]))
            zz = np.expand_dims(zz, 0)
            temp = tuple([self.parm['state_size'][0] * self.parm['state_size'][1]])
            summary(self.actor.to(self.gpu), temp)
            self.writer.add_graph(self.actor, torch.tensor(zz).to(self.gpu).float())

            zz = np.zeros((self.parm['state_size'][0] * self.parm['state_size'][1]+self.action_size))
            zz = np.expand_dims(zz, 0)
            temp = tuple([self.parm['state_size'][0] * self.parm['state_size'][1]+self.action_size])
            summary(self.critic.to(self.gpu), temp)
            self.writer.add_graph(self.critic, torch.tensor(zz).to(self.gpu).float())

        else:
            zz = np.zeros((self.parm['state_size']))
            zz = np.expand_dims(zz, 0)
            summary(self.agent.to(self.gpu), tuple(self.parm['state_size']))
            self.writer.add_graph(self.agent, torch.tensor(zz).float().to(self.gpu))
        self.noise = OUNoise(action_space(self.parm['action_size']),theta=self.parm['theta'],max_sigma=
                             self.parm['sigma'],min_sigma=0,decay_period=self.parm['run_step'])
        self.t = 0
        self.clip_cr = self.optimizer_parms['critic']['clipping']=='True'
        self.clip_ac = self.optimizer_parms['actor']['clipping']=='True'

    def generate_optimizer(self):

        actor_optimizer = get_optimizer(self.optimizer_parms['actor'], self.actor)
        critic_optimizer = get_optimizer(self.optimizer_parms['critic'], self.critic)

        return actor_optimizer, critic_optimizer

    def get_action(self, state):
        if ~torch.is_tensor(state):
            state = torch.tensor(state).to(self.gpu).float()

        size = [-1]
        for i in self.state_size:
            size.append(i)
        state = state.view(size)
        self.agent = self.agent.eval()
        action_mean,value = self.agent.forward(state)
        self.t +=1
        action = self.noise.get_action(action_mean.to(self.cpu).detach().numpy(),t=self.t)
        return action

    def zero_grad(self):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

    def train(self):
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        self.agent = self.agent.train()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append([mini_batch[i][1]])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])
        actions = torch.tensor(actions).view(self.batch_size,self.action_size).to(self.gpu).float()
        __, target = self.target_agent.forward(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
            else:
                target[i] = rewards[i] + self.discount_factor * target[i]

        loss_policy, loss_critic = self.agent.loss(states, target,actions)
        self.zero_grad()
        loss_policy.backward()
        self.critic.zero_grad()
        loss_critic.backward()
        critic_norm = calculate_global_norm(self.critic)
        self.critic_optimizer.step()
        self.actor_optimizer.step()



        return loss_policy+loss_critic, critic_norm