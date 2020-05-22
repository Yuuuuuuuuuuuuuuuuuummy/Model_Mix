import torch
import torch.nn as nn


class ModuleA(nn.Module):
    def __init__(self):
        super(ModuleA, self).__init__()
        self.Conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.Max_pool = nn.MaxPool2d(kernel_size=9, stride=1)
        self.Mid_Conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.Final_Conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.Conv(x)
        out = self.Mid_Conv(out)
        out = self.Final_Conv(self.Max_pool(out))
        return out.squeeze(dim=2).squeeze(dim=2)


class ModuleB(nn.Module):
    def __init__(self):
        super(ModuleB, self).__init__()
        self.Conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.Max_pool = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        out = self.Conv(x)# [B, 1, 3, 3]
        out = self.Max_pool(out) #[B, 1, 1, 1]
        return out.squeeze(dim=2).squeeze(dim=2)


class ModuleC(nn.Module):
    def __init__(self):
        super(ModuleC, self).__init__()
        self.Conv_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.Conv_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.Maxpool = nn.MaxPool2d(kernel_size=5, stride=1)

    def forward(self, x):
        out = self.Conv_0(x)#[B, 3, 5, 5]
        out = self.Conv_1(out) # [B, 1, 5, 5]
        out = self.Maxpool(out) # [B, 1, 1, 1]
        return out.squeeze(dim=2).squeeze(dim=2)