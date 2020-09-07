import torch
import torch.nn as nn
from baseline.utils import find_activation, Flatten, Nomramlization






class Base_Agent(nn.Module):

    def __init__(self):
        super(Base_Agent, self).__init__()

    def build_model(self):
        pass

    def forward(self,state,*args):
        pass

    def loss(self, *args):
        pass
class MLP(nn.Module):

    def __init__(self,init_size=1,num_of_layers=1, num_of_unit=32,activation='relu',**kwargs):

        super(MLP, self).__init__()
        self.batch_normal = False
        self.normal = False

        if 'normal' in kwargs.keys():
            self.normal = kwargs['normal']

        if 'batch_norm' in kwargs.keys():
            self.batch_normal =  kwargs['batch_norm']
        self.init_size = init_size
        self.num_of_layers = num_of_layers
        if type(num_of_unit) == int:
            num_of_unit = [num_of_unit]
        self.num_of_unit = list(num_of_unit)
        self.activation = activation
        self.module = self.build_model()

    def build_model(self):

        model = nn.Sequential()
        init_size = self.init_size
        if self.normal:
            model.add_module('normal', self.normal)
        for i in range(self.num_of_layers):
            model.add_module('layer_'+str(i), torch.nn.Linear(init_size, self.num_of_unit[i]))
            init_size = self.num_of_unit[i]

            if self.batch_normal:
                model.add_module("batch_norm_"+str(i), nn.BatchNorm1d(self.num_of_unit[i]))
            if self.activation != 'linear':
                feature_act = find_activation(self.activation)
                model.add_module('act_'+str(i), feature_act)
        return model

    def forward(self,state):

        return self.module(state)


class ConvNet(nn.Module):

    def __init__(self, init_size=3,num_of_layers=1, num_of_unit=32,kernel_size = 1, stride = 1, padding = 0,activation='relu',**kwargs):
        super(ConvNet, self).__init__()


        if 'normalization' in kwargs.keys():
            self.normalization = kwargs['normalization']
        if 'flatten' in kwargs.keys():
            self.flatten = kwargs['flatten']
        if 'batch_norm' in kwargs.keys():
            self.batch_normal = kwargs['batch_norm']



        self.init_size = init_size
        self.num_of_layers = num_of_layers
        self.num_of_unit = list(num_of_unit)
        self.kernel_size = list(kernel_size)
        self.stride = list(stride)
        self.padding = list(padding)
        self.activation = activation

        self.module = self.build_model()


    def build_model(self):

        model = nn.Sequential()
        init_size = self.init_size
        if self.normalization:
            model.add_module('normalization', Nomramlization())
        for i in range(self.num_of_layers):
            model.add_module('layer_'+str(i),nn.Conv2d(init_size, self.num_of_unit[i],self.kernel_size[i], self.stride[i],
                                                       padding=self.padding[i]))
            init_size = self.num_of_unit[i]
            feature_act = find_activation(self.activation)

            if self.batch_normal:
                model.add_module('batch_norm_'+str(i), nn.BatchNorm2d(self.num_of_unit[i]))


            model.add_module('act+' + str(i), feature_act)

        if self.flatten:
            model.add_module('flatten', Flatten())

        return model
