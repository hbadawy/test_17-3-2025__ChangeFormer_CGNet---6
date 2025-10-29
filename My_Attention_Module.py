
import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8, device=None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False, device=device)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False, device=device)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out = (self.avg_pool(x))
        # print ("avg out :  ", avg_out.shape)
        avg_out = (self.fc1(avg_out))
        # print ("avg out :  ", avg_out.shape)
        avg_out = self.fc2(self.relu1(avg_out))


        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)




class SpatialAttention(nn.Module):
    def __init__(self, device=None):
        super(SpatialAttention,self).__init__()

        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        # print (avg_out.shape)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]
        # print (max_out.shape)

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)
    


class My_SpatialAttention(nn.Module):
    def __init__(self, device=None):
        super(My_SpatialAttention,self).__init__()

        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guiding_map0):

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        # print ("guiding_map0", guiding_map0.shape)
        guiding_map = F.sigmoid(guiding_map0)
        # print ("guid map max:   ", guiding_map.max(), "guid map min:   ", guiding_map.min())


        avg_out = torch.mean(x,dim=1,keepdim=True)
        # print (avg_out.shape)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]
        # print (max_out.shape)

        avg_out_guided = avg_out * guiding_map
        max_out_guided = max_out * guiding_map

        # x = torch.cat([avg_out,max_out],dim=1)
        x = torch.cat([avg_out_guided,max_out_guided],dim=1)

        x = self.conv1(x)
        
        return self.sigmoid(x)
    


################## My Idea ######################

class My_GuideModule(nn.Module):
    def __init__(self, in_channels, device=None):
        super(My_GuideModule,self).__init__()

        self.in_channels = in_channels
        self.ca = ChannelAttention(self.in_channels)
        self.my_sa = My_SpatialAttention ()

    def forward(self, x, guiding_map0):
        skip = x
        x = x*self.ca(x)
        x = x*self.my_sa(x, guiding_map0)
        x = x+skip
        return x
    

if __name__ == "__main__":
    # x = torch.rand(1,3,256, 256)
    # gmap = torch.rand(1,1,16,16)
    # model = My_SpatialAttention()
    # y = model(x, gmap)
    # # x = x*model(x)
    # print (y.shape)


    x = torch.rand(1,8,256, 256)
    gmap = torch.rand(1,1,16,16)
    model = My_GuideModule(8)
    y = model(x, gmap)
    # x = x*model(x)
    print (y.shape)