# 导入Pytorch相关的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 定义ResNet18的模型类


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        # 使用torchvision中的预训练模型
        self.resnet18 = models.resnet18(pretrained=True)
        # 替换最后一层为自定义的全连接层
        self.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        # 通过ResNet18的卷积层和池化层
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        # 平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 展平
        x = torch.flatten(x, 1)
        # 通过自定义的全连接层
        x = self.fc(x)
        return x