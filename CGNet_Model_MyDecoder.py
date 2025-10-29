
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from CGNet.ChangeGuideModule import *
from ChangeFormerV4_MyEncoder import *

################################################################################################
######################### Encorder of ChangeFormer1 + CGNet Decoder ############################
################################################################################################

class CGNet(nn.Module):
    def __init__(self, device=None):
        super(CGNet, self).__init__()
        # vgg16_bn = models.vgg16_bn(pretrained=True)
        # self.inc = vgg16_bn.features[:5]  # 64
        # self.down1 = vgg16_bn.features[5:12]  # 128
        # self.down2 = vgg16_bn.features[12:22]  # 256
        # self.down3 = vgg16_bn.features[22:32]  # 512
        # self.down4 = vgg16_bn.features[32:42]  # 512

        self.Tenc = EncoderTransformer_x3(device=device)

        self.conv_reduce_1 = BasicConv2d(32*2,32,3,1,1, device=device) 
        self.conv_reduce_2 = BasicConv2d(64*2,64,3,1,1, device=device)
        self.conv_reduce_3 = BasicConv2d(128*2,128,3,1,1, device=device)
        self.conv_reduce_4 = BasicConv2d(256*2,256,3,1,1, device=device)

        self.up_layer4 = BasicConv2d(512,512,3,1,1, device=device)
        self.up_layer3 = BasicConv2d(512,512,3,1,1, device=device)
        self.up_layer2 = BasicConv2d(256,256,3,1,1, device=device)

        self.decoder = nn.Sequential(BasicConv2d(256,64,3,1,1, device=device),nn.Conv2d(64,1,3,1,1, device=device))

        self.decoder_final = nn.Sequential(BasicConv2d(32,16,3,1,1, device=device),nn.Conv2d(16,1,1, device=device))

        self.cgm_2 = ChangeGuideModule(64, device=device)
        self.cgm_3 = ChangeGuideModule(128, device=device)
        self.cgm_4 = ChangeGuideModule(256, device=device)

        #相比v2 额外的模块   Additional modules compared to v2
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(384,128,3,1,1, device=device)
        self.decoder_module3 = BasicConv2d(192,64,3,1,1, device=device)
        self.decoder_module2 = BasicConv2d(96,32,3,1,1, device=device)

        self.sigmoid = nn.Sigmoid()

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A

    def forward(self,A,B):

        size = A.size()[2:]

        [fx1, fx2] = [self.Tenc(A), self.Tenc(B)]

        # print (len(fx1))  # 4

        layer1_A = fx1[0]  #; print ("layer1_A:  ", layer1_A.shape)  # [1, 32, 128, 128]
        layer2_A = fx1[1]  #; print ("layer2_A:  ", layer2_A.shape)  # [1, 64, 64, 64]
        layer3_A = fx1[2]  #; print ("layer3_A:  ", layer3_A.shape)  # [1, 128, 32, 32]
        layer4_A = fx1[3]  #; print ("layer4_A:  ", layer4_A.shape)  # [1, 256, 16, 16]

        layer1_B = fx2[0]
        layer2_B = fx2[1]
        layer3_B = fx2[2]
        layer4_B = fx2[3]



        layer1 = torch.cat((layer1_B,layer1_A),dim=1)  #; print ("layer1:  ", layer1.shape)  # [1, 64, 128, 128]

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)  #; print ("layer2:  ", layer2.shape)  # [1, 128, 64, 64]

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)  #; print ("layer3:  ", layer3.shape)  # [1, 256, 32, 32]

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)  #; print ("layer4:  ", layer4.shape)  # [1, 512, 16, 16]

        layer1 = self.conv_reduce_1(layer1)   #; print ("layer1 after conv_reduce_1:  ", layer1.shape)  # [1, 32, 128, 128]
        layer2 = self.conv_reduce_2(layer2)   #; print ("layer2 after conv_reduce_2:  ", layer2.shape)  # [1, 64, 64, 64]
        layer3 = self.conv_reduce_3(layer3)   #; print ("layer3 after conv_reduce_3:  ", layer3.shape)  # [1, 128, 32, 32]
        layer4 = self.conv_reduce_4(layer4)   #; print ("layer4 after conv_reduce_4:  ", layer4.shape)  # [1, 256, 16, 16]
        # print ("layer1:   ", layer1.shape)  # 1, 128, 128, 128
        # print ("layer2:   ", layer2.shape)  # 1, 256, 64, 64
        # print ("layer3:   ", layer3.shape)  #[1, 512, 32, 32]       
        # print ("layer4:   ", layer4.shape)  #[1, 512, 16, 16]

        # layer4 = self.up_layer4(layer4)
        #
        # layer3 = self.up_layer3(layer3)
        #
        # layer2 = self.up_layer2(layer2)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse=layer4_1              # ; print ("feature_fuse:  ", feature_fuse.shape)  #[1, 256, 128, 128]
        # print ("feature_fuse:  ", feature_fuse.shape)  #[1, 512, 128, 128])


        # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        # layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        # feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)

        change_map = self.decoder(feature_fuse)             #; print ("change map shape after decoder: ", change_map.shape)  #[1, 1, 128, 128]
        # print ("change map shape: ", change_map.shape)  #[1, 1, 128, 128]

   


        layer4 = self.cgm_4(layer4, change_map)     #; print ("layer4 after cgm4:  ", layer4.shape)   #[1, 256, 16, 16]
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))     #; print ("feature4 after decoder module4:   ", feature4.shape)  # [1, 128, 32, 32]
        # print ("feature4:   ", feature4.shape)  # [1, 512, 32, 32]

        layer3 = self.cgm_3(feature4, change_map)    #; print ("layer3 after cgm3:  ", layer3.shape)  #[[1, 128, 32, 32]
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3),layer2],1))   #; print ("feature3 after decoder module3:   ", feature3.shape)  # [1, 64, 64, 64]

        layer2 = self.cgm_2(feature3, change_map)    #; print ("layer2 after cgm2:  ", layer2.shape)  #[1, 64, 64, 64]
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))   #; print ("layer1 after decoder module2:   ", layer1.shape)  # [1, 32, 128, 128]
        # print ("layer1 after last cat and dec: ", layer1.shape)  #[1, 128, 128, 128]

        #change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)   #[1, 1, 256, 256]

        final_map = self.decoder_final(layer1)    #; print ("final_map:   ", final_map.shape)  #[1, 1, 128, 128]
        # print ("final_map:   ", final_map.shape)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)    #; print ("final_map after interpolate:   ", final_map.shape)  #[1, 1, 256, 256]
        # print (final_map.min(), final_map.max())  # tensor(-1.1961, grad_fn=<MinBackward1>) tensor(1.7514, grad_fn=<MaxBackward1>)

        final_map = self.sigmoid(final_map)
        # print (final_map.min(), final_map.max())
        return final_map #, change_map
    




if __name__ == "__main__":

        # model = models.vgg16_bn(pretrained=True)
        # import torchinfo
        # torchinfo.summary(model, input_size=[(1,3,256,256)])


        x1 = torch.rand([1, 3,256, 256])
        x2 = torch.rand([1, 3,256, 256])
        # # print ("x1 size: " , x1.size()[2:])
        # # print ("x1 min: ", x1.min(), "x1 max: ", x1.max())
        # # print ("x mean:", x1.mean(), "x std: ", x1.std())
        model = CGNet()
        y1 = model(x1, x2)
        print (y1.shape)   #[1, 8,256, 256]


        import torchinfo
        torchinfo.summary(model, input_size=[(1,3,256,256), (1,3,256,256)])