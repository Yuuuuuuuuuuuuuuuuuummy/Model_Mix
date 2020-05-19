import torch
import numpy as np
import torch.nn as nn
from model_ import ModuleA
from torchviz import make_dot


def main():
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
    stat = {'model_state_dict':state_dict}
    torch.save(stat, 'model_A.pth')

    state = torch.load('model_A.pth', map_location='cpu')
    print(state['model_state_dict']['Conv.weight'])



if __name__ == '__main__':
    main()