import torch.nn as nn
import torch
import torch.nn.functional as F
from mytransformer import RandomResizedCrop,Resize


def randompinjie(images,data_trans):
    if data_trans=='train':
        for ind in range(images.shape[0]):
            if ind==0:
                a=RandomResizedCrop(224)(images[ind,:,:,:])
                a=torch.unsqueeze(a, dim=0)#维度扩充，因为attention输出是N 64 100 709
            else:
                c=RandomResizedCrop(224)(images[ind,:,:,:])
                c=torch.unsqueeze(c, dim=0)
                a=torch.cat([a,c],dim=0)
    elif data_trans=='val':
        for ind in range(images.shape[0]):
            if ind==0:
                a=Resize([224, 224])(images[ind,:,:,:])
                a=torch.unsqueeze(a, dim=0)#维度扩充，因为attention输出是N 64 100 709
            else:
                c=Resize([224, 224])(images[ind,:,:,:])
                c=torch.unsqueeze(c, dim=0)
                a=torch.cat([a,c],dim=0)
        
    return a

class selectfunction(nn.Module):
    #img图像
    #weight权重矩阵
    #num提取主要特征个数
    #name控制全局选取还是每个图片选取
    #每张图片权重排序并选出主要特征 

    def __init__(self, num,name):
        super(selectfunction, self).__init__()
        self.num=num
        self.name=name

    def forward(self, img,weight):
        if self.name=='single':  
            target=torch.zeros(img.shape[0],img.shape[1],img.shape[2],self.num)
            for i in range(img.shape[0]):
                ind=weight[i,:,0,0].argsort()[-self.num:]
                target[i,:,:,:]=img[i,:,:,ind]
        #batch中每张图片权重相加然后排序选出主要特征
        if self.name=='global':
            ind=weight[:,:,0,0].sum(0).argsort()[-self.num:]#权重相加后，前100特征索引
            target=img[:,:,:,ind]
        return target

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)        
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class SE(nn.Module):

    def __init__(self, in_chnls, mid_chnls):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, mid_chnls, 1, 1, 0)
        self.excitation = nn.Conv2d(mid_chnls, in_chnls, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mid_chnls)
        self.bn2 = nn.BatchNorm2d(in_chnls)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = self.bn2(out)
        return torch.sigmoid(out)

class Select_Attention(nn.Module):
    
    def __init__(self,select_num,select_name,in_chnls1=709, mid_chnls1=45,in_channel=1,out_channel=64):
        super(Select_Attention, self).__init__()
        self.block=BasicBlock(in_channel,out_channel)#channel 的输入输出
        self.se1=SE(in_chnls1, mid_chnls1)
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1, bias=False) 
        self.relu = nn.ReLU()
        self.bn=nn.BatchNorm2d(out_channel)
        self.sele=selectfunction(select_num,select_name)
     
    def forward(self, x):
        x=self.block(x)#[16,64,100,709]
        out1=x.permute(0,3,1,2)#[16,64,100,709]->[16,709,64,100]
        coefficient1=self.se1(out1)#权重矩阵[16,709,1,1]
  
        x= self.sele(x,coefficient1).cuda()

        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)
        
        return x


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 select_num=100, 
                 select_name='single',
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.atten=Select_Attention(select_num,select_name)
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(64, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x,data_trans):
        x= self.atten(x)
        x = randompinjie(x,data_trans)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2],select_num=select_num,select_name=select_name,num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3],select_num=select_num,select_name=select_name,num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3],select_num=select_num,select_name=select_name,num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3],select_num=select_num,select_name=select_name,num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnet152-b121ed2d.pth
    return ResNet(Bottleneck, [3, 8, 36, 3],select_num=select_num,select_name=select_name,num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  select_num=select_num,
                  select_name=select_name,
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000,select_num=100,select_name='single' ,include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  select_num=select_num,
                  select_name=select_name,
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
