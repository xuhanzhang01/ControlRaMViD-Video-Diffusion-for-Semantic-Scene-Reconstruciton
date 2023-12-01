import torch
import yaml

checkpoint = torch.load('models/seg_re_model240000.pt')

for key in checkpoint.keys():
    print(key)
