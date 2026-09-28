# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Mnist12(torch.nn.Module):
    def __init__(self):
        super(Mnist12, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Mnist12::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #Mnist12::Mnist12/Conv2d[conv1]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #Mnist12::Mnist12/ReLU[relu1]/189
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Mnist12::Mnist12/MaxPool2d[pool1]/input.5
        self.module_4 = py_nndct.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #Mnist12::Mnist12/Conv2d[conv2]/input.7
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #Mnist12::Mnist12/ReLU[relu2]/223
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[3, 3], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #Mnist12::Mnist12/MaxPool2d[pool2]/237
        self.module_7 = py_nndct.nn.Module('flatten') #Mnist12::Mnist12/input
        self.module_8 = py_nndct.nn.Linear(in_features=256, out_features=10, bias=True) #Mnist12::Mnist12/Linear[fc]/241

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(input=output_module_0, start_dim=1, end_dim=3)
        output_module_0 = self.module_8(output_module_0)
        return output_module_0
