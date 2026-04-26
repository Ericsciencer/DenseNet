import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """DenseNet的基本层：Bottleneck结构 (BN-ReLU-Conv1x1-BN-ReLU-Conv3x3)"""
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        # 保存输入用于后续拼接
        prev_features = x
        # Bottleneck前向传播
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        # 拼接输入与当前层输出
        return torch.cat([prev_features, x], dim=1)


class DenseBlock(nn.Module):
    """堆叠多个DenseLayer形成Dense Block"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(
                    in_channels + i * growth_rate,  # 每层输入通道数递增
                    growth_rate,
                    bn_size,
                    drop_rate
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """过渡层：降维（1x1卷积）+ 下采样（2x2平均池化）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate=32,          # 增长率k（每层输出通道数）
        block_config=(6,12,24,16), # 每个Dense Block的层数（DenseNet-121配置）
        num_init_features=64,    # 初始卷积层输出通道数
        bn_size=4,                # Bottleneck中1x1卷积的通道倍数
        drop_rate=0.0,            # Dropout率
        num_classes=1000          # 分类数
    ):
        super().__init__()

        # 1. 初始卷积层 + 最大池化
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. 堆叠Dense Block与Transition Layer
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 添加Dense Block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f"denseblock{i+1}", block)
            # 更新通道数（Block输出通道数 = 输入 + k*层数）
            num_features += num_layers * growth_rate
            # 最后一个Block后不加Transition Layer
            if i != len(block_config) - 1:
                # 添加Transition Layer（压缩率θ=0.5）
                trans = TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # 3. 最后的BN + ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # 4. 分类头：全局平均池化 + 全连接
        self.classifier = nn.Linear(num_features, num_classes)

        # 权重初始化（He初始化）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))  # 全局平均池化到1x1
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121(num_classes=1000, **kwargs):
    return DenseNet(block_config=(6,12,24,16), num_classes=num_classes, **kwargs)

def densenet169(num_classes=1000, **kwargs):
    return DenseNet(block_config=(6,12,32,32), num_classes=num_classes, **kwargs)

def densenet201(num_classes=1000, **kwargs):
    return DenseNet(block_config=(6,12,48,32), num_classes=num_classes, **kwargs)

def densenet264(num_classes=1000, **kwargs):
    return DenseNet(block_config=(6,12,64,48), num_classes=num_classes, **kwargs)

if __name__ == "__main__":
    # 测试DenseNet-121
    model = densenet121(num_classes=10)
    # 随机输入：(batch_size, channels, height, width)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")  # 应为 (2, 10)