import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from fpn import FPN50
from retinanet import RetinaNet


d=torch.load('./model/resnet50.pth')
fpn=FPN50()
dd=fpn.state_dict()
# for k in d.keys():
#     print(k)
    # if not k.startswith('fc'):  # skip fc layers
    #     dd[k] = d[k]

net=RetinaNet()

# for m in net.modules():
#     print(m)

print(net.cls_head)