import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """Custom SE block compatible with torchvision versions that lack 'reduction' arg"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    """Enhanced convolutional block with residual connection and SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels, reduction=16) if use_se else nn.Identity()
        self.relu = nn.SiLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += identity
        return self.relu(out)

class LungCancerCNN(nn.Module):
    """Modern CNN architecture with attention mechanisms"""
    def __init__(self, num_classes=2):  # Changed from 3 to 2
        super().__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        # Feature blocks
        self.block1 = self._make_layer(64, 64, 2, stride=1)
        self.block2 = self._make_layer(64, 128, 2, stride=2)
        self.block3 = self._make_layer(128, 256, 3, stride=2)
        self.block4 = self._make_layer(256, 512, 3, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ConvBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)       # 64x192x192
        x = self.block1(x)     # 64x192x192
        x = self.block2(x)     # 128x96x96
        x = self.block3(x)     # 256x48x48
        x = self.block4(x)     # 512x24x24
        
        x = self.avgpool(x)    # 512x1x1
        x = torch.flatten(x, 1) # 512
        x = self.dropout(x)
        x = self.fc(x)         # num_classes
        
        return x

    def get_cam_target_layers(self):
        """Return layers suitable for Grad-CAM visualization"""
        return [self.block4[-1]]
