import torch
import torch.nn as nn


# 归一化函数
class MySoftMax(nn.Module):
    def __init__(self):
        super(MySoftMax, self).__init__()

    def forward(self, x):
        x = x - torch.min(x) + 0.01
        return x / torch.sum(x)


# 交叉熵损失函数
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        return -torch.sum(y * torch.log(x + 1e-10))


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y, t):
        return -t*torch.sum(y * torch.log(x + 1e-10))


# 避免梯度消失和梯度爆炸交叉熵损失函数
class CrossEntropyLoss_noNaN(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_noNaN, self).__init__()

    def forward(self, x, y):
        return -torch.sum((y + 1e-10) * torch.log2(x + 1e-10))


# 折半交叉熵损失函数
class HalfCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(HalfCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        return -torch.sum((x + y) / 2 * torch.log(x + 1e-10))


# 混合型损失函数
class MixedLoss(nn.Module):
    def __init__(self, num=2):
        super(MixedLoss, self).__init__()
        self.label_num = num

    def forward(self, x, y, t):
        a = 1/self.label_num
        return -t * torch.sum(y * torch.log(x + 1e-10)) + (1-t) * torch.sum((x-a)**2)


# 软目标损失函数1
class SoftenLoss(nn.Module):
    def __init__(self, label_num=2):
        super(SoftenLoss, self).__init__()
        self.a = 1 / label_num

    def forward(self, x):
        return torch.sum((x - self.a) ** 2)


# 软目标损失函数2
class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, y, t):
        y = self.soft(y/t)
        return -torch.sum(y * torch.log(x + 1e-10))
