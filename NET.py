# from models import SELayer,ChannelAttention,SpatialAttention
from torch import nn
import torch
import torch.nn.functional as F
from ECA.eca_resnet import ECABottleneck

# from functions import CrissCrossAttention
# from part import *


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class DoubleConvM(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16) -> object:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.PReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.shape[0],-1))
        avg_out2 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc(self.max_pool(x).view(x.shape[0],-1))
        max_out2 = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out.view(x.shape[0], -1, 1, 1)+avg_out2+max_out2)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.pad=nn.ReplicationPad2d(padding)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x=self.pad(x)
        x = self.conv1(x)
        return self.sigmoid(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, channels, or23=2):
        super(Conv, self).__init__()
        self.or23 = or23
        if or23 == 2:
            self.conv=nn.Sequential(
                nn.ReplicationPad2d(2),
                nn.Conv2d(channels, channels, kernel_size=5, padding=0),
                nn.ReplicationPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(inplace=True),
                nn.ReplicationPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReplicationPad3d(2),
                nn.Conv3d(1, 3, kernel_size=5, padding=0),
                nn.ReplicationPad3d(1),
                nn.Conv3d(3, 3, kernel_size=3, padding=0),
                nn.BatchNorm3d(3),
                nn.LeakyReLU(inplace=True),
                nn.ReplicationPad3d(1),
                nn.Conv3d(3, 1, kernel_size=3, padding=0),
                # nn.BatchNorm3d(1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.or23 == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        if self.or23 == 3:
            x = torch.squeeze(x, dim=1)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, shape):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AdaptiveMaxPool2d(shape),
            DoubleConvM(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
# class interpolation(nn.Module):
#     def __init__(self):
#         super(interpolation, self).__init__()
#         self.interpolate=torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=None)
#
#     def forward(self, x):
#         x.transpose_([0,2,3,1])
#         x = self.interpolate(x)
#         x.transpose_([0, 3, 1, 2])
#         return x



class SCABlock(nn.Module):
    def __init__(self, channels):
        super(SCABlock, self).__init__()
        self.pconv2d = Conv(channels, 2)
        # self.pconv3d = Conv(channels, 3)
        # self.ChannelAttention = ChannelAttention(channels, ratio=10)
        self.SpatialAttention = SpatialAttention(kernel_size=3)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(inplace=True)
        self.eca_layer = ECABottleneck(channels, channels)

    def forward(self, x, res):

        # x_r = x
        x2d = self.pconv2d(x)
        # x3d = self.pconv3d(x)
        x = x2d
        xc = self.eca_layer(x)
        # x = xc
        x_r = xc
        x2d = self.pconv2d(x)
        # x3d = self.pconv3d(x)
        x = x2d+res
        # xc = x
        xs = self.SpatialAttention(x)
        x = self.prelu(xs)
        return self.relu(xs.expand_as(x) * x_r)


# class RCCAModule(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes):
#         super(RCCAModule, self).__init__()
#         inter_channels = in_channels // 4
#         self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(inter_channels))
#         self.cca = CrissCrossAttention(inter_channels)
#         self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                    nn.BatchNorm2d(inter_channels))
#
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
#             )
#
#     def forward(self, x, recurrence=1):
#         output = self.conva(x)
#         for i in range(recurrence):
#             output = self.cca(output)
#         output = self.convb(output)
#
#         output = self.bottleneck(torch.cat([x, output], 1))
#         return output


class RA_SubNet(nn.Module):
    def __init__(self, n_channels, n_classes, mid_channels, CRN = 2, cz=False):
        super(RA_SubNet, self).__init__()
        if cz==True:
            n_channels = 31
        self.inc = DoubleConvM(n_channels, mid_channels)
        # self.ChannelAttention = ChannelAttention(mid_channels, ratio=int(CRN*mid_channels//n_classes))
        # self.SpatialAttention = SpatialAttention(kernel_size=3)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()
        self.outc = OutConv(mid_channels, n_classes)
        self.pconv2d = Conv(mid_channels, 2)
        # self.pconv3d = Conv(mid_channels, 3)
        self.dblock = SCABlock(mid_channels)
        self.cz = cz
        # self.cca = RCCAModule(mid_channels, mid_channels, mid_channels)
        # self.interpolation = interpolation()

    def forward(self, x):
        if self.cz == True:
            x = torch.transpose(x, 1, 3)
            x = torch.nn.functional.interpolate(x, size=[241,128], scale_factor=None, mode='bilinear', align_corners=False)
            x = torch.transpose(x, 1, 3)
        x = self.prelu(x)
        res = x = self.inc(x)
        # x = self.cca(x)
        x_r = self.pconv2d(x)
        x0 = self.dblock(x_r, x)
        x1 = self.dblock(x0, x_r)
        x2 = self.dblock(x1, x0)
        x3 = self.dblock(x2, x1)
        x4 = self.dblock(x3, x2)
        x5 = x4+res
        return self.outc(x5)

class PTNet(nn.Module):
    def __init__(self, n_channels, n_classes, mid_channels,size:tuple,CRN = 2 ,cz=False):
        super(PTNet, self).__init__()
        self.size = size
        self.RA_SubNet0 = RA_SubNet(mid_channels,mid_channels//2,mid_channels, CRN, cz)
        self.RA_SubNet12 = RA_SubNet(size[0]//2, size[0]//2, size[0]//2, CRN, False)
        self.RA_SubNet13 = RA_SubNet(size[1]//2, size[1]//2, size[1]//2, CRN, False)
        self.mish = Mish()
        self.prelu = nn.PReLU()
        self.head = DoubleConvM(n_channels, mid_channels)
        self.tail = DoubleConvM(mid_channels, mid_channels)
        # self.tailcca = RCCAModule(mid_channels, mid_channels, n_classes*2)
        self.Down = Down(mid_channels,mid_channels,[size[0]//2,size[1]//2])
        self.outc = OutConv(mid_channels, n_classes)
        self.outc12 = OutConv( mid_channels, mid_channels//2)
        self.outc13 = OutConv( mid_channels, mid_channels//2)
        self.con = DoubleConvM(int(1.5*mid_channels), mid_channels)
        self.con2 = DoubleConvM(mid_channels//2, mid_channels//2)
        self.bn = nn.BatchNorm2d(int(1.5*mid_channels))

    def forward(self, x):
        x = self.prelu(x)
        res = x = self.head(x)
        x = self.Down(x)
        x12 = x.permute([0,2,1,3])
        x13 = x.permute([0,3,2,1])
        x = self.RA_SubNet0(res)
        x12 = self.RA_SubNet12(x12)
        x12 = x12.permute([0,2,1,3])
        x13 = self.RA_SubNet13(x13)
        x13 = x13.permute([0,3,2,1])
        x12 = self.outc12(x12)
        x13 = self.outc13(x13)
        x12 = torch.nn.functional.interpolate(x12, size=[self.size[0], self.size[1]], scale_factor=None, mode='bilinear',
                                            align_corners=False)
        x13 = torch.nn.functional.interpolate(x13, size=[self.size[0], self.size[1]], scale_factor=None, mode='bilinear',
                                            align_corners=False)
        x12 = self.con2(x12)
        x13 = self.con2(x13)
        x = torch.cat((x,x12,x13),1)
        # x = torch.nn.functional.interpolate(x, size=[self.size[0], self.size[1]], scale_factor=None, mode='bilinear', align_corners=False)
        x = self.bn(x)
        x = self.con(x)
        x = self.tail(x+res)
        result = self.outc(x)
        return result
