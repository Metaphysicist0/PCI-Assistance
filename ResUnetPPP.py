import torch
import torch.nn as nn

'''
From:
ResUNet++: An Advanced Architecture for Medical Image Segmentation 
author={D. {Jha} and P. H. {Smedsrud} and M. A. {Riegler} and D. {Johansen} and T. D. {Lange} and P. {Halvorsen} and H. {D. Johansen}},
booktitle={2019 IEEE International Symposium on Multimedia (ISM)}, 
'''

import torch.nn as nn
import torch.nn.functional as F


class DecomNet(nn.Module):
    def __init__(self, num_layers):
        super(DecomNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding=4))
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img = torch.cat((torch.max(input=img, dim=1, keepdim=True)[0], img), dim=1)
        output = self.model(img)
        R, I = output[:, :3, :, :], output[:, 3:4, :, :]
        return R, I


class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(True)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu3 = nn.ReLU(True)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu2 = nn.ReLU(True)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(True)

        self.fusion_conv = nn.Conv2d(in_channels=321, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, R, I):
        # enhance的输入是两个
        I = torch.cat((R, I), dim=1)
        h1 = self.conv1(I)
        x = self.relu1(h1)

        h2 = self.conv2(x)
        x = self.relu2(h2)

        h3 = self.conv3(x)

        h2_ = self.up_conv3(h3)
        h2_ = torch.cat((h2, h2_), dim=1)
        x = self.up_relu3(h2_)

        h1_ = self.up_conv2(x)
        h1_ = torch.cat((h1, h1_), dim=1)
        x = self.up_relu2(h1_)

        x = self.up_conv1(x)
        x = self.activation(x)

        c1 = nn.UpsamplingNearest2d(scale_factor=2)(h1_)
        c2 = nn.UpsamplingNearest2d(scale_factor=4)(h2_)
        c3 = nn.UpsamplingNearest2d(scale_factor=8)(h3)

        x = torch.cat([x, c1, c2, c3], dim=1)

        x = self.fusion_conv(x)
        x = self.final_conv(x)
        # x = self.final_activation(x)

        return x


class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CA_Block(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CA_Block, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),

            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            CA_Block(out_c, out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
            CA_Block(out_c, out_c)
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y


class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y


class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d


class RetinexNetResUNetPP(nn.Module):

    def __init__(self, num_layers):
        super().__init__()
# Initialize
        self.c1 = Stem_Block(3, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)

        self.b1 = ASPP(128, 256)

        self.d1 = Decoder_Block([64, 256], 128)
        self.d2 = Decoder_Block([32, 128], 64)
        self.d3 = Decoder_Block([16, 64], 32)

        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 3, kernel_size=1, padding=0)
        # 可视化改3 测时改1

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(True)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu3 = nn.ReLU(True)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu2 = nn.ReLU(True)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(True)

        self.fusion_conv = nn.Conv2d(in_channels=321, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

# RetinexNet
    # Decom_Net
        layers = []
        layers.append(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding=4))

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img = torch.cat((torch.max(input=img, dim=1, keepdim=True)[0], img), dim=1)
        output = self.model(img)
        R, I = output[:, :3, :, :], output[:, 3:4, :, :]
        return R, I

    # Enhance_Net
    def forward(self, R, I):
        # enhance的输入是两个
        I = torch.cat((R, I), dim=1)
        h1 = self.conv1(I)
        x = self.relu1(h1)

        h2 = self.conv2(x)
        x = self.relu2(h2)

        h3 = self.conv3(x)

        h2_ = self.up_conv3(h3)
        h2_ = torch.cat((h2, h2_), dim=1)
        x = self.up_relu3(h2_)

        h1_ = self.up_conv2(x)
        h1_ = torch.cat((h1, h1_), dim=1)
        x = self.up_relu2(h1_)

        x = self.up_conv1(x)
        x = self.activation(x)

        c1 = nn.UpsamplingNearest2d(scale_factor=2)(h1_)
        c2 = nn.UpsamplingNearest2d(scale_factor=4)(h2_)
        c3 = nn.UpsamplingNearest2d(scale_factor=8)(h3)

        x = torch.cat([x, c1, c2, c3], dim=1)

        x = self.fusion_conv(x)
        x = self.final_conv(x)
        # x = self.final_activation(x)

        return x

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.aspp(d3)
        output = self.output(output)

        return output


if __name__ == "__main__":

    net = RetinexNetResUNetPP(5)