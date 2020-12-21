import math
from torch.nn import functional as F
from functools import partial
import torch.nn as nn
import torch


################################################## Same Padding Conv ####################################################

class SamePadConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)


########################################### Swish Activation Function #################################################

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


############################################# conv-bn-active Module ###################################################

def conv_bn_swish(in_channels, out_channels,
                  kernel_size, stride=1,
                  groups=1, active=True,
                  eps=1e-3, momentum=0.01,
                  bias=True,image_size=32):
    if active:
        return nn.Sequential(
            SamePadConv2d(in_channels=in_channels, out_channels=out_channels,
                   groups=groups, kernel_size=kernel_size,
                   stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
            Swish()
        )
    else:
        return nn.Sequential(
            SamePadConv2d(in_channels=in_channels, out_channels=out_channels,
                   groups=groups, kernel_size=kernel_size,
                   stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        )


################################################## SE Module ##########################################################

class SEModule(nn.Module):
    """ squeeze and excitation module for channel attention """

    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

################################################# Drop Connect ########################################################

class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()


################################################### MB Conv ###########################################################

class MB_conv(nn.Module):
    """' MB conv is the basic module of efficient network."""

    def __init__(self, in_channels, out_channels,
                 expand, kernel_size, stride,
                 se_ratio=0.25, drop_connect_ratio=0.2,
                 image_size=32):
        super().__init__()
        expand_channels = in_channels * expand
        # 1 × 1 conv for expanding
        self.expand_conv = conv_bn_swish(in_channels, expand_channels,
                                         kernel_size=1, bias=False,
                                         image_size=image_size)
        # depth wise convolution
        self.dw_conv = conv_bn_swish(expand_channels, expand_channels,
                                     kernel_size=kernel_size, bias=False,
                                     image_size=image_size)
        # squeeze and expand module (channel attention)
        self.se = SEModule(expand_channels, int(in_channels * se_ratio))
        # conv for outputting
        self.project_conv = conv_bn_swish(expand_channels, out_channels,
                                          kernel_size=1, bias=False,
                                          active=False, image_size=image_size)
        # skip and drop-connect
        self.skip = (stride == 1) and (in_channels == out_channels)
        # self.drop_connect = DropConnect(drop_connect_ratio)
        self.drop_connect = nn.Identity()

    def forward(self, x):
        x_input = x
        x = self.expand_conv(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.drop_connect(x)
            x = x + x_input
        return x


################################################### MB Block ##########################################################

class MB_block(nn.Module):
    """'MB block is a repeated accumulation of MB conv.
        Efficient network is consist of mann MB block with different parameters."""

    def __init__(self, repeat_num, in_channels,
                 out_channels, expand, kernel_size,
                 stride, se_ratio=0.25, drop_connect_ratio=0.2,
                 image_size=32):
        super().__init__()
        layers = [MB_conv(in_channels, out_channels, expand,
                          kernel_size, stride, se_ratio=se_ratio,
                          drop_connect_ratio=drop_connect_ratio,
                          image_size=image_size)]
        for i in range(1, repeat_num):
            layers.append(MB_conv(out_channels, out_channels, expand,
                                  kernel_size, stride=1, se_ratio=se_ratio,
                                  drop_connect_ratio=drop_connect_ratio,
                                  image_size=image_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


############################################### Efficient Network #####################################################

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, image_size=32):
        super().__init__()
        # stem
        self.stem = conv_bn_swish(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False, image_size=image_size)
        # backbone
        self.blocks = nn.Sequential(
            MB_block(repeat_num=1, in_channels=32, out_channels=16, expand=1, kernel_size=3, stride=1, image_size=image_size),
            MB_block(repeat_num=2, in_channels=16, out_channels=24, expand=6, kernel_size=3, stride=2, image_size=image_size),
            MB_block(repeat_num=2, in_channels=24, out_channels=40, expand=6, kernel_size=5, stride=2, image_size=image_size),
            MB_block(repeat_num=3, in_channels=40, out_channels=80, expand=6, kernel_size=3, stride=2, image_size=image_size),
            MB_block(repeat_num=3, in_channels=80, out_channels=112, expand=6, kernel_size=5, stride=1, image_size=image_size),
            MB_block(repeat_num=4, in_channels=112, out_channels=192, expand=6, kernel_size=5, stride=2, image_size=image_size),
            MB_block(repeat_num=1, in_channels=192, out_channels=320, expand=6, kernel_size=3, stride=1, image_size=image_size)
        )
        # head
        self.head = nn.Sequential(
            conv_bn_swish(in_channels=320, out_channels=1280, kernel_size=1, bias=False, image_size=image_size),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(p=0.2, inplace=True)
        )
        # linear layer
        self.linear = nn.Linear(1280,num_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


