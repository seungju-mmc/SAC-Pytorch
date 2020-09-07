import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.baseline_agent import Base_Agent, MLP, ConvNet
from baseline.utils import get_output_size

class DDPG_Agent(Base_Agent):

    def __init__(self, parms):
        super(DDPG_Agent, self).__init__()
        self.parms = parms
        self.batch_norm = False
        if self.parms['network'] == 'MLP':
            self.init_size = self.parms['state_size'][0] * self.parms['state_size'][-1]
            self.critic_init_size = self.init_size + self.parms['action_size']
            self.mode = 'MLP'
        else:
            self.init_size = self.parms['state_size'][0]
            self.normalization = self.parms['normalization'] == 'True'
            if 'batch_norm' in self.parms.keys():
                self.batch_norm = self.parms['batch_norm'] == 'True'

            self.kernel_size = list(self.parms['kernel_size'])
            self.padding = list(self.parms['padding'])
            self.stride = list(self.parms['stride'])

            self.mode = 'Conv'

        self.action_size = self.parms['action_size']
        self.device = torch.device(self.parms['device'])

        self.actor_parm = self.parms['actor']
        self.a_n_layers = self.actor_parm['num_of_layers']
        self.a_n_unit = self.actor_parm['filter_size']
        self.a_act = self.actor_parm['activation']
        self.a_l_act = self.actor_parm['actor_activation']
        self.a_bn = self.actor_parm['batch_norm']=='True'

        self.critic_parm = self.parms['critic']
        self.c_n_layers = self.critic_parm['num_of_layers']
        self.c_act = self.critic_parm['activation']
        self.c_n_unit = self.critic_parm['filter_size']
        self.c_bn = self.critic_parm['batch_norm']=='True'

        self.input_size = [-1]
        if self.mode == 'MLP':
            self.input_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][1]]
            self.critic_input_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][1]
                                      +self.parms['action_size']]
        elif self.mode == 'Conv':
            for i in self.parms['state_size']:
                self.input_size.append(i)

        self.build_model()

    def build_model(self):
        self.actor = nn.Sequential()

        if self.mode == 'MLP':

            self.actor_Feature = MLP(init_size=self.init_size,
                               num_of_layers=self.a_n_layers,
                               num_of_unit=self.a_n_unit,
                               activation=self.a_act,
                                     batch_norm=self.a_bn).module

            shape = get_output_size(self.actor_Feature.eval(),self.input_size.copy())

            self.l_actor = MLP(init_size=shape[-1],
                                     num_of_layers=1,
                                     num_of_unit=self.parms['action_size'],
                                     activation=self.a_l_act,
                               batch_norm=self.a_bn).module

            self.critic_ = MLP(init_size=self.critic_init_size,
                                     num_of_layers=self.c_n_layers,
                                     num_of_unit=self.c_n_unit,
                                     activation=self.c_act,
                               batch_norm=self.c_bn).module



        elif self.mode == 'Conv':
            self.Feature = ConvNet(init_size=self.init_size,
                                   num_of_layers=self.num_of_layers,
                                   num_of_unit=self.num_of_unit,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   flatten=True,
                                   batch_norm=self.batch_norm
                                   ).module


        self.actor.add_module('Embedding',self.actor_Feature)
        self.actor.add_module('actor',self.l_actor)
        self.actor = self.actor.to(self.device)
        if self.c_bn:
            self.critic = nn.Sequential(*list(self.critic_.children())[:-2]).to(self.device)
        else:
            self.critic = nn.Sequential(*list(self.critic_.children())[:-1]).to(self.device)
    def forward(self, state):

        if torch.is_tensor(state) == False:
            state = torch.tensor(state).to(self.device).float()

        state = state.view(self.input_size).to(self.device).float()

        actor = self.actor(state)
        concat = torch.cat((state,actor), dim=1)
        critic = self.critic(concat)

        return actor, critic

    def loss(self, states, targets,actions):
        if torch.is_tensor(states) == False:
            states = torch.tensor(states).to(self.device).float()
        states = states.view(self.input_size).to(self.device).float()
        embedding = torch.cat((states,actions),dim=1)
        critic = self.critic(embedding)

        targets = targets.to(self.device).float()

        loss_critic = torch.mean((critic-targets).pow(2))
        policy, critic = self.forward(states)
        loss_policy = -torch.mean(critic)


        return loss_policy, loss_critic