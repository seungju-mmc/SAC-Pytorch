import torch
import random
import numpy as np

from baseline.baseline_trainier import Policy
from config_DQN.Agent import DQN_Agent
from baseline.utils import get_optimizer
from torchsummary import summary

class DQN_Trainer(Policy):

    def __init__(self, file_name):
        super(DQN_Trainer,self).__init__(file_name)
        if self.Exploration_method == 'e_greedy':
            self.epsilon = self.parm['epsilon']

        if self.inference_mode:
            self.epsilon = 0
            self.epsilon_min = 0

        self.agent =DQN_Agent(self.agent_parms)
        self.target_agent = DQN_Agent(self.agent_parms)
        self.optimizer = self.generate_optimizer()

        if len(self.parm['state_size']) < 3:
            zz = np.zeros((self.parm['state_size'][0] * self.parm['state_size'][1]))
            zz = np.expand_dims(zz, 0)
            temp = tuple([self.parm['state_size'][0] * self.parm['state_size'][1]])
            summary(self.agent.to(self.gpu), temp)
            self.writer.add_graph(self.agent, torch.tensor(zz))

        else:
            zz = np.zeros((self.parm['state_size']))
            zz = np.expand_dims(zz, 0)
            summary(self.agent.to(self.gpu), tuple(self.parm['state_size']))
            self.writer.add_graph(self.agent, torch.tensor(zz).float().to(self.gpu))

    def get_action(self, state):
        if ~torch.is_tensor(state):
            state = torch.tensor(state)

        size = [-1]
        for i in self.state_size:
            size.append(i)
        state = state.view(size)

        if self.Exploration_method == 'e_greedy':
            if self.epsilon > np.random.rand():
                return np.random.randint(0, self.action_size)
            else:
                Q = self.agent.forward(state)
                a = torch.argmax(Q).detach().to(self.cpu)
                return np.int(a.numpy())

        elif self.Exploration_method == 'boltzman':
            Q = self.agent.forward(state)
            Q = Q.detach().to(self.cpu)
            Q = torch.softmax(Q, dim=-1).squeeze().numpy()
            action = [i for i in range(self.action_size)]
            a = np.random.choice(action, p=Q)

            return a

    def generate_optimizer(self):
        return get_optimizer(self.optimizer_parms,self.agent)

    def train(self):
        mini_batch = random.sample(self.replay_memory, self.batch_size)

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

        target = [torch.max(self.target_agent.forward(next_state).detach()) for next_state in next_states]
        actions_ = np.zeros((self.batch_size, self.action_size))

        for i in range(self.batch_size):
            if dones[i]:
                target[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
            else:
                target[i] = rewards[i] + self.discount_factor * target[i]
            a = np.array(actions[i])
            a = a.reshape(1)
            actions_[i, a[0]] = 1
        loss = self.agent.loss(states, target, actions_)
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in self.agent.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = (total_norm) ** (1. / 2)
        self.optimizer.step()

        return loss, total_norm


