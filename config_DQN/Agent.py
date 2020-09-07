import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.baseline_agent import Base_Agent, MLP, ConvNet
from baseline.utils import get_output_size

class DQN_Agent(Base_Agent):

    def __init__(self, parms):
        super(DQN_Agent, self).__init__()
        self.parms = parms

        self.batch_norm = False
        if self.parms['network'] =='MLP':
            self.init_size = self.parms['state_size'][0]*self.parms['state_size'][-1]
            self.mode = 'MLP'
        else:
            self.init_size = self.parms['state_size'][0]
            self.normalization = self.parms['normalization'] =='True'
            if 'batch_norm' in self.parms.keys():
                self.batch_norm = self.parms['batch_norm']=='True'


            self.kernel_size = list(self.parms['kernel_size'])
            self.padding = list(self.parms['padding'])
            self.stride = list(self.parms['stride'])

            self.mode='Conv'

        self.num_of_layers = self.parms['num_of_layers']
        self.num_of_unit = list(self.parms['filter_size'])
        self.activation = self.parms['activation']
        self.fully_connected_size = self.parms['fully_connected_size']

        self.action_size = self.parms['action_size']
        self.device = torch.device(self.parms['device'])
        self.dueling = self.parms['dueling']=='True'

        self.input_size = [-1]
        if self.mode == 'MLP':
            self.input_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][1]]
        elif self.mode == 'Conv':
            for i in self.parms['state_size']:
                self.input_size.append(i)

        self.build_model()


    def build_model(self):
        self.DQN = nn.Sequential()

        if self.mode =='MLP':

            self.Feature = MLP(init_size=self.init_size,
                        num_of_layers=self.num_of_layers,
                        num_of_unit=self.num_of_unit,
                        activation=self.activation).module

        elif self.mode =='Conv':
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

        output_size = get_output_size(self.Feature, self.input_size.copy())
        self.fully_feature = MLP(init_size=output_size[-1],
                                 num_of_layers=1,
                                 num_of_unit=self.fully_connected_size).to(self.device).module
        if self.dueling:
            self.adv = MLP(init_size=self.fully_connected_size,
                      num_of_layers=1,
                      num_of_unit=self.action_size).to(self.device)
            self.adv = nn.Sequential(*list(self.adv.module.children())[:-1])
            self.v = MLP(init_size=self.fully_connected_size,
                    num_of_layers=1,
                    num_of_unit=1).to(self.device)
            self.v = nn.Sequential(*list(self.v.module.children())[:-1])
        else:
            self.q = MLP(init_size=self.fully_connected_size,
                      num_of_layers=1,
                      num_of_unit=self.action_size).to(self.device).module
            self.q = nn.Sequential(*list(self.q.children())[:-1])

        self.Feature = self.Feature.to(self.device)
        self.fully_feature = self.fully_feature.to(self.device)


    def forward(self,state):

        if torch.is_tensor(state)==False:
            state = torch.tensor(state).to(self.device).float()

        state = state.view(self.input_size).to(self.device).float()

        if self.dueling:
            f = self.Feature(state)
            k = self.fully_feature(f)
            adv = self.adv(k)
            v = self.v(f)
            q = adv+v-adv.mean(dim=1, keepdim=True)
        else:
            f = self.Feature(state)
            k = self.fully_feature(f)
            q = self.q(k)

        return q

    def loss(self, states, targets, actions):
        values = self.forward(states)

        if torch.is_tensor(actions) == False:
            actions = torch.tensor(actions).to(self.device).float()

        selected_values = torch.sum(values * actions, dim=1)
        targets = torch.stack(targets)

        selected_values = selected_values.view((-1, 1))
        targets = targets.view((-1, 1))

        loss = F.smooth_l1_loss(selected_values, targets)

        return loss




