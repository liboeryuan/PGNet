import torch
import torch.nn as nn
import torch.nn.functional as F

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None


    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
  
class RFB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )
        self.conv = nn.Conv2d(4 * out_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x + x1)
        x3 = self.conv3(x + x2)

        out = self.conv(torch.cat([x, x1, x2, x3], dim=1))
        return out



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self, ch1, ch2):
        super(EAM, self).__init__()
        # self.reduce1 = Conv1x1(256, 64)
        # self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(ch1 + ch2, ch2, 3),
            ConvBNR(ch2, ch2, 3),
            nn.Conv2d(ch2, 1, 1))

    def forward(self, x1, x4):
        size = x1.size()[2:]
        # x1 = self.reduce1(x1)
        # x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class edge_att(nn.Module):
    def __init__(self, ch):
        super(edge_att, self).__init__()
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        edge = self.mlp_shared(edge)
        
        out = x * edge + x
        return out


class CFI(nn.Module):
    def __init__(self, hchannel, channel):
        super(CFI, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 2, channel // 2, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 2, channel // 2, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.dconv6_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=5)
        self.dconv4_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=7)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, hf, lf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 2, dim=1)
        x11 = self.conv3_1(xc[0] + xc[1])
        x12 = self.dconv7_1(x11 + xc[1])

        x21 = torch.chunk(x11, 2, dim=1)
        x22 = torch.chunk(x12, 2, dim=1)

        x211 = self.dconv5_1(x21[0] + x21[1])
        x212 = self.dconv9_1(x21[1] + x211 + x22[0])

        x221 = self.dconv6_1(x212 + x22[0] + x22[1])
        x222 = self.dconv4_1(x221 + x22[1])

        xx = self.conv1_2(torch.cat((x211, x212, x221, x222), dim=1))
        x = self.conv3_3(x + xx)

        return x 

class Net(nn.Module):
    def __init__(self, in_ch=1, dim=32, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [dim, dim*2, dim*4, dim*8, dim*16]
        self.maxpool = nn.MaxPool2d(2, 2)
        self.e1 = Res_CBAM_block(in_ch, filters[0])
        self.e2 = Res_CBAM_block(filters[0], filters[1])
        self.e3 = Res_CBAM_block(filters[1], filters[2])
        self.e4 = Res_CBAM_block(filters[2], filters[3])
        self.e5 = Res_CBAM_block(filters[3], filters[4])

        self.edge = EAM(filters[1], filters[4])

        self.att5 = RFB(filters[4], filters[4])
        self.att4 = RFB(filters[3], filters[3])
        self.att3 = RFB(filters[2], filters[2])
        self.att2 = RFB(filters[1], filters[1])
        self.att1 = RFB(filters[0], filters[0])

        self.edgeatt5 = edge_att(filters[4])
        self.edgeatt4 = edge_att(filters[3])
        self.edgeatt3 = edge_att(filters[2])
        self.edgeatt2 = edge_att(filters[1])
        self.edgeatt1 = edge_att(filters[0])

        self.cfi4 = CFI(filters[4], filters[3])
        self.cfi3 = CFI(filters[3], filters[2])
        self.cfi2 = CFI(filters[2], filters[1])
        self.cfi1 = CFI(filters[1], filters[0])

        
        self.final_conv5 = nn.Conv2d(filters[4], 1, 3, 1, 1)
        self.final_conv4 = nn.Conv2d(filters[3], 1, 3, 1, 1)
        self.final_conv3 = nn.Conv2d(filters[2], 1, 3, 1, 1)
        self.final_conv2 = nn.Conv2d(filters[1], 1, 3, 1, 1)
        self.final_conv1 = nn.Conv2d(filters[0], 1, 3, 1, 1)
        

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.maxpool(e1)
        e2 = self.e2(e2)
        e3 = self.maxpool(e2)
        e3 = self.e3(e3)
        e4 = self.maxpool(e3)
        e4 = self.e4(e4)
        e5 = self.maxpool(e4)
        e5 = self.e5(e5)

        edge = self.edge(e2, e5)

        e5 = self.att5(e5)
        e4 = self.att4(e4)
        e3 = self.att3(e3)
        e2 = self.att2(e2)
        e1 = self.att1(e1)

        e5 = self.edgeatt5(e5, edge.sigmoid())
        e4 = self.edgeatt4(e4, edge.sigmoid())
        e3 = self.edgeatt3(e3, edge.sigmoid())
        e2 = self.edgeatt2(e2, edge.sigmoid())
        e1 = self.edgeatt1(e1, edge.sigmoid())
       
        d4 = self.cfi4(e5, e4)
        d3 = self.cfi3(d4, e3)
        d2 = self.cfi2(d3, e2)
        d1 = self.cfi1(d2, e1)

        
        pred5 = self.final_conv5(e5)
        pred4 = self.final_conv4(d4)
        pred3 = self.final_conv3(d3)
        pred2 = self.final_conv2(d2)
        pred1 = self.final_conv1(d1)
        predict5 = F.interpolate(pred5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(pred4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(pred3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(pred2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(pred1, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge = F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict5.sigmoid(), predict4.sigmoid(), predict3.sigmoid(), predict2.sigmoid(), predict1.sigmoid(), edge.sigmoid()
        
        
if __name__ == "__main__":
    x = torch.rand(4, 1, 256, 256)
    net = Net()
    out = net(x)
    for i in out:
        print(i.shape)

