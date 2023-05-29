# encoding: utf-8

import torch.nn as nn
import torch.nn.functional as F
import torch

# 用于数字序列的残差块   
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        #print(out.size())##################################################3
        return out
    
# 用于图片的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, inchannel, C, kernel_size, stride, padding=1, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super(PoolBN, self).__init__()
        self.conv1 = self.left = nn.Sequential(
            nn.Conv1d(inchannel, inchannel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(inchannel*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(inchannel*2, inchannel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(inchannel*2),
            nn.ReLU(inplace=True),
        )

        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        elif  pool_type == 'conv':
            self.pool = self.left = nn.Sequential(
                nn.Conv1d(inchannel*2, inchannel*2, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(inchannel*2),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out
        



####################################################################################################       

# GAN生成器
class Generator(nn.Module):
    def __init__(self, inputlength = 1801, n_residual_blocks=5):
        super(Generator, self).__init__()

        #输入数据点的个数
        self.inputlength = inputlength
        self.data_length = inputlength

        # 初始卷积块     
        self.premodel = nn.Sequential(
                    nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.ReLU(),
                    nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU()
                    ) 

        # 下采样 Downsampling
        # 初始卷积块卷积后通道数为16
        in_features = 16
        out_features = in_features*2

        model = []
        # 加入两个降采样模块
        for _ in range(2):
            self.data_length = (self.data_length - 1)//2 + 1
            model += [  nn.Conv1d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        #经过两次降采样，加通道降长度，out_features = 16 * 4

        # 残差块 Residual blocks
        for _ in range(n_residual_blocks):
            
            model += [ResBlock(in_features, in_features)]

        # 上采样 Upsampling
        out_features = in_features//2 #in_features = out_features
        for _ in range(2):
            self.data_length = (self.data_length - 1)*2 +1
            # nn.ConvTranspose1d
            # 输出长度 = (输入长度 - 1) * 步长 - 2*填充 + 卷积核大小k
            model += [  nn.ConvTranspose1d(in_features, out_features, 3, stride=2, padding=1, output_padding=0),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        self.model = nn.Sequential(*model)

        #self.BN = nn.BatchNorm1d(inputLength)

        # 输出层 Output layer
        self.fc = nn.Linear(16 * self.data_length, self.inputlength)

    def forward(self, x):
        y = x
        out = x.view(-1, 1, self.inputlength)
        out = self.premodel(out)#([64, 16, 1301])
        #print(out.shape)
        out = self.model(out)#([64, 16, 1301])
        #print(out.shape)
        out = out.view(out.size(0), -1)#([64, 20816])
        #print(out.shape)
        out = self.fc(out)#torch.Size([64, 1301])

        #out = 0.9*out + 0.1*y
        #out = torch.sigmoid(out)
        #print(out.shape)

        return out

# 判别器
class Discriminator(nn.Module):
    def __init__(self, inputlength = 1801):
        super(Discriminator, self).__init__()

        self.inputlength = inputlength

        self.model = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                    )
        
        self.ResBlock1 = ResBlock(inchannel=32, outchannel=64)
        self.Poolmax = PoolBN('max', 64, C=2, kernel_size=3, stride=2)
        self.ResBlock2 = ResBlock(inchannel=128, outchannel=64)
        self.ResBlock3 = ResBlock(inchannel=64, outchannel=128)

        self.ResBlock4 = ResBlock(inchannel=128, outchannel=64)
        self.ResBlock5 = ResBlock(inchannel=64, outchannel=128)
        
        self.fc = nn.Linear(in_features=128 * (inputlength//4), out_features=1)

    def forward(self, x):
        out = x.view(-1, 1, self.inputlength)

        out = self.model(out)
        out = self.ResBlock1(out)
        out = self.Poolmax(out)
        #print(out.size)
        out = self.ResBlock2(out) 
        out = self.ResBlock3(out) #[16,128,325]
        out = self.ResBlock4(out)
        out = self.ResBlock5(out)
        out = out.view(-1, 128 * ( self.inputlength//4 ))
        
        out = self.fc(out)
        #out = torch.sigmoid(out)
        
        return out#输出(batchsize,1)
    

