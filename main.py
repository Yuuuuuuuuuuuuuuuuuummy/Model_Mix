import torch
import numpy as np
import torch.nn as nn
from model_ import ModuleA, ModuleB, ModuleC
import collections
from torchviz import make_dot

def A():
    input_A = torch.rand((1, 3, 9, 9)).requires_grad_(True)
    output_A = torch.rand((1, 1)).requires_grad_(True)
    model_A = ModuleA()
    model_A.train()
    loss_function = nn.L1Loss()
    optim = torch.optim.Adam(model_A.parameters(), lr=0.001)
    output = model_A(input_A)
    loss = loss_function(output, output_A)
    optim.zero_grad()
    loss.backward()
    optim.step()

    input_A = torch.rand((1, 3, 9, 9)).requires_grad_(True)
    output_A = torch.rand((1, 1)).requires_grad_(True)
    output = model_A(input_A)
    loss = loss_function(output, output_A)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # graph = make_dot(output.mean(), params=dict(list(model_A.named_parameters()) + [('input', input_A)]))
    # graph.view()

    state_dict = model_A.state_dict()
    # print(state_dict['Conv.weight'])
    stat = {'model_state_dict': state_dict}
    torch.save(stat, 'model_A.pth')

    # state = torch.load('model_A.pth', map_location='cpu')
    # print(state['model_state_dict']['Conv.weight'])


def B():
    input_B = torch.rand((1, 3, 3, 3)).requires_grad_(True)
    output_B = torch.rand((1, 1)).requires_grad_(True)
    model_B = ModuleB()
    model_B.train()
    loss_function = nn.L1Loss()
    optim = torch.optim.Adam(model_B.parameters(), lr=0.001)
    output = model_B(input_B)
    loss = loss_function(output, output_B)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # graph = make_dot(output.mean(), params=dict(list(model_B.named_parameters()) + [('input', input_B)]))
    # graph.view()

    input_B = torch.rand((1, 3, 3, 3)).requires_grad_(True)
    output_B = torch.rand((1, 1)).requires_grad_(True)
    output = model_B(input_B)
    loss = loss_function(output, output_B)
    optim.zero_grad()
    loss.backward()
    optim.step()

    state_dict = model_B.state_dict()
    print(state_dict['Conv.weight'])
    stat = {'model_state_dict': state_dict}
    torch.save(stat, 'model_B.pth')

    state = torch.load('model_B.pth', map_location='cpu')
    print(state['model_state_dict']['Conv.weight'])


def C(a_path, b_path):
    input_C = torch.rand((1, 3, 5, 5)).requires_grad_(True)
    output_C = torch.rand((1, 1)).requires_grad_(True)
    model_C = ModuleC()
    output = model_C(input_C)
    # graph = make_dot(output.mean(), params=dict(list(model_C.named_parameters()) + [('input', input_C)]))
    # graph.view()
    a_stat = torch.load(a_path)
    b_stat = torch.load(b_path)
    # print('A:', a_stat['model_state_dict'])
    # print('B:', b_stat['model_state_dict'])
    mix_stat = collections.OrderedDict()
    mix_stat['Conv_0.weight'] = a_stat['model_state_dict']['Conv.weight']
    mix_stat['Conv_0.bias'] = a_stat['model_state_dict']['Conv.bias']
    mix_stat['Conv_1.weight'] = b_stat['model_state_dict']['Conv.weight']
    mix_stat['Conv_1.bias'] = b_stat['model_state_dict']['Conv.bias']
    print(mix_stat)
    model_C.load_state_dict(mix_stat)


def main():
    # A()
    # B()
    C('model_A.pth', 'model_B.pth')

if __name__ == '__main__':
    main()