import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck

class Part_body_stream(nn.Module):
    def __init__(self, body_model_path, body_model_name, pretrain_choice):
        super(Part_body_stream, self).__init__()
        if body_model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif body_model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif body_model_name == 'resnet18-bsize':
            self.in_planes = 512
            self.base = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif body_model_name == 'resnet34-bsize':
            self.in_planes = 512
            self.base = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])

        self.gap = nn.AdaptiveAvgPool2d((1, 8))

        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap6 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap7 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap8 = nn.AdaptiveAvgPool2d((1, 1))


        self.num_part = 8

        if pretrain_choice == 'imagenet':
            self.base.load_param(body_model_path)
            print('Loading pretrained ImageNet model(res-net18/res-net34)......')

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        # x = self.relu(x)                # add missed relu
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)           # 只取前三层， feature map为256

        # print(x.shape,'---------------')
        # part = {}
        # for i in range(self.num_part):
        #     part[0] = self.gap1(x[:,:,:,i])


        body_feat = self.gap(x)
        part = {}
        #get eight part feature
        for i in range(self.num_part):
            part[i] = body_feat[:, :, :, i]                    #batch,channel,height,width
        body_feature = torch.cat((part[0], part[1], part[2], part[3], part[4], part[5], part[6], part[7]), dim=1)  #256×8=2048
        body_feature = body_feature.view(body_feature.shape[0], -1)  # flatten to (bs, 2048)
        return body_feature

class Part_paw_stream(nn.Module):
    def __init__(self, body_model_path, body_model_name, pretrain_choice):
        super(Part_paw_stream, self).__init__()
        if body_model_name == 'resnet18':
            self.in_planes = 512
            self.base1 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base2 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base3 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base4 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base5 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base6 = ResNet(last_stride=2,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
        elif body_model_name == 'resnet34':
            self.in_planes = 512
            self.base1 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base2 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base3 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base4 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base5 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base6 = ResNet(last_stride=2,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif body_model_name == 'resnet18-bsize':
            self.in_planes = 512
            self.base1 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base2 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base3 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base4 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base5 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
            self.base6 = ResNet(last_stride=1,
                            block=BasicBlock,
                            layers=[2, 2, 2, 2])
        elif body_model_name == 'resnet34-bsize':
            self.in_planes = 512
            self.base1 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base2 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base3 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base4 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base5 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base6 = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])

        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))

        if pretrain_choice == 'imagenet':
            self.base1.load_param(body_model_path)
            print('Loading 1 pretrained ImageNet model(res-net18/res-net34)......')
            self.base2.load_param(body_model_path)
            print('Loading 2 pretrained ImageNet model(res-net18/res-net34)......')
            self.base3.load_param(body_model_path)
            print('Loading 3 pretrained ImageNet model(res-net18/res-net34)......')
            self.base4.load_param(body_model_path)
            print('Loading 4 pretrained ImageNet model(res-net18/res-net34)......')
            self.base5.load_param(body_model_path)
            print('Loading 5 pretrained ImageNet model(res-net18/res-net34)......')
            self.base6.load_param(body_model_path)
            print('Loading 6 pretrained ImageNet model(res-net18/res-net34)......')

    def forward(self, x):
        #传进来的x应该是6个部分的图片在通道维度上cat形成
        x1 = x[:, 0:3, :, :]                            #左前
        x2 = x[:, 3:6, :, :]                            #右前
        x3 = x[:, 6:9, :, :]                            #右后上
        x4 = x[:, 9:12, :, :]                           #右后下
        x5 = x[:, 12:15, :, :]                          #左后上
        x6 = x[:, 15:18, :, :]                          #左后下

        parts_x = [x1, x2, x3, x4, x5, x6]
        part_feat = {}
        for i, x in enumerate(parts_x):
            if i == 0:
                x = self.base1.conv1(x)
                x = self.base1.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base1.maxpool(x)
                x = self.base1.layer1(x)
                x = self.base1.layer2(x)
                x = self.base1.layer3(x)
                part_feat[i+1] = x
            if i == 1:
                x = self.base2.conv1(x)
                x = self.base2.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base2.maxpool(x)
                x = self.base2.layer1(x)
                x = self.base2.layer2(x)
                x = self.base2.layer3(x)
                part_feat[i+1] = x
            if i == 2:
                x = self.base3.conv1(x)
                x = self.base3.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base3.maxpool(x)
                x = self.base3.layer1(x)
                x = self.base3.layer2(x)
                x = self.base3.layer3(x)
                part_feat[i+1] = x
            if i == 3:
                x = self.base4.conv1(x)
                x = self.base4.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base4.maxpool(x)
                x = self.base4.layer1(x)
                x = self.base4.layer2(x)
                x = self.base4.layer3(x)
                part_feat[i+1] = x
            if i == 4:
                x = self.base5.conv1(x)
                x = self.base5.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base5.maxpool(x)
                x = self.base5.layer1(x)
                x = self.base5.layer2(x)
                x = self.base5.layer3(x)
                part_feat[i+1] = x
            if i == 5:
                x = self.base6.conv1(x)
                x = self.base6.bn1(x)
                # x = self.relu(x)                # add missed relu
                x = self.base6.maxpool(x)
                x = self.base6.layer1(x)
                x = self.base6.layer2(x)
                x = self.base6.layer3(x)
                part_feat[i+1] = x

        #对两条后腿的特征融合，由6部分特征变为4部分
        behind_top = torch.add(part_feat[3], part_feat[5])
        behind_down = torch.add(part_feat[4], part_feat[6])

        new_parts_x = [part_feat[1], part_feat[2], behind_top, behind_down]
        new_part_feat = {}
        for i, x in enumerate(new_parts_x):
            if i == 0:
                x = self.base1.layer4(x)
                x = self.gap1(x)
                new_part_feat[i+1] = x
            if i == 1:
                x = self.base2.layer4(x)
                x = self.gap2(x)
                new_part_feat[i+1] = x
            if i == 2:
                x = self.base3.layer4(x)
                x = self.gap3(x)
                new_part_feat[i+1] = x
            if i == 3:
                x = self.base4.layer4(x)
                x = self.gap4(x)
                new_part_feat[i+1] = x

        paw_feature = torch.cat((new_part_feat[1], new_part_feat[2], new_part_feat[3], new_part_feat[4]), dim=1)
        paw_feature = paw_feature.view(paw_feature.shape[0], -1)  # flatten to (bs, 2048)
        return paw_feature