import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Building Blocks
class ConvBnRelu(nn.Module):
    """Convolution + BatchNorm + ReLU module"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block with two ConvBnRelu layers"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Dropout2d(dropout_rate),
            ConvBnRelu(out_channels, out_channels),
            nn.Dropout2d(dropout_rate)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        return F.relu(x + residual)

class SelfAttention(nn.Module):
    """Self-attention module"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(b, -1, h * w)
        value = self.value(x).view(b, -1, h * w)

        scale = (c // 8) ** -0.5
        energy = torch.bmm(query, key) * scale
        attention = F.softmax(energy, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x

class EncoderBlock(nn.Module):
    """Encoder block with optional self-attention"""
    def __init__(self, in_channels, out_channels, use_attention=False, dropout_rate=0.2):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, dropout_rate)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x):
        x = self.res_block(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class DecoderBlock(nn.Module):
    """Decoder block with skip connections and optional self-attention"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False, dropout_rate=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels // 2 + skip_channels, out_channels, dropout_rate)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class SAMFeatureProcessor(nn.Module):
    """Generate multi-scale SAM features"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.level3_conv = ConvBnRelu(in_channels, 256)
        self.level4_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.level2_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.level1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        level3 = self.level3_conv(x)  # [B, 256, 64, 64]
        level4 = self.level4_conv(x)  # [B, 512, 32, 32]

        level2_feat = self.level2_conv(x)  # [B, 128, 64, 64]
        level1_feat = self.level1_conv(x)  # [B, 64, 64, 64]
        level2 = F.interpolate(level2_feat, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, 128, 128]
        level1 = F.interpolate(level1_feat, scale_factor=4, mode='bilinear', align_corners=True)  # [B, 64, 256, 256]

        return {
            'level1': level1,
            'level2': level2,
            'level3': level3,
            'level4': level4
        }

# Simple feature fusion without gating/CBAM for ablation
class FeatureFusionSimple(nn.Module):
    """Feature fusion module without gating and CBAM (ablation)"""
    def __init__(self, image_channels, mask_channels, sam_channels, out_channels):
        super().__init__()
        self.image_conv = ConvBnRelu(image_channels, out_channels // 2)
        self.mask_conv = ConvBnRelu(mask_channels, out_channels // 2)
        self.sam_conv = ConvBnRelu(sam_channels, out_channels)
        self.output_conv = ConvBnRelu(out_channels, out_channels)

    def forward(self, image_feat, mask_feat, sam_feat):
        proc_image = self.image_conv(image_feat)
        proc_mask = self.mask_conv(mask_feat)
        regular_feat = torch.cat([proc_image, proc_mask], dim=1)

        proc_sam = self.sam_conv(sam_feat)
        if proc_sam.shape[2:] != regular_feat.shape[2:]:
            proc_sam = F.interpolate(
                proc_sam,
                size=regular_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        fused = regular_feat + proc_sam
        out = self.output_conv(fused)
        return out

class ResUNet(nn.Module):
    """RefineNet variant without gating & CBAM for ablation study"""
    def __init__(self, num_classes=17, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes

        # 初始处理
        self.image_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.mask_processor = nn.Sequential(
            nn.Conv2d(17, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.sam_processor = SAMFeatureProcessor(256)

        # 编码器
        self.image_encoder1 = EncoderBlock(64, 64, use_attention=False, dropout_rate=dropout_rate)
        self.image_encoder2 = EncoderBlock(64, 128, use_attention=False, dropout_rate=dropout_rate)
        self.image_encoder3 = EncoderBlock(128, 256, use_attention=True, dropout_rate=dropout_rate)
        self.image_encoder4 = EncoderBlock(256, 512, use_attention=True, dropout_rate=dropout_rate)

        self.mask_encoder1 = EncoderBlock(64, 64, use_attention=False, dropout_rate=dropout_rate)
        self.mask_encoder2 = EncoderBlock(64, 128, use_attention=False, dropout_rate=dropout_rate)
        self.mask_encoder3 = EncoderBlock(128, 256, use_attention=True, dropout_rate=dropout_rate)
        self.mask_encoder4 = EncoderBlock(256, 512, use_attention=True, dropout_rate=dropout_rate)

        # 特征融合（去掉门控/CBAM）
        self.fusion1 = FeatureFusionSimple(64, 64, 64, 64)
        self.fusion2 = FeatureFusionSimple(128, 128, 128, 128)
        self.fusion3 = FeatureFusionSimple(256, 256, 256, 256)
        self.fusion4 = FeatureFusionSimple(512, 512, 512, 512)

        # 瓶颈
        self.bottleneck = ResidualBlock(512, 1024, dropout_rate=dropout_rate)

        # 解码器
        self.decoder4 = DecoderBlock(1024, 512, 512, use_attention=True, dropout_rate=dropout_rate)
        self.decoder3 = DecoderBlock(512, 256, 256, use_attention=True, dropout_rate=dropout_rate)
        self.decoder2 = DecoderBlock(256, 128, 128, use_attention=False, dropout_rate=dropout_rate)
        self.decoder1 = DecoderBlock(128, 64, 64, use_attention=False, dropout_rate=dropout_rate)

        # 最终输出
        self.final_conv = nn.Sequential(
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, images, teeth_mask, sam_embedding):
        """Forward pass
        Args:
            images: [B, 3, H, W] 原始图像
            teeth_mask: [B, 17, H, W] 牙齿掩码
            sam_embedding: [B, 256, 64, 64] SAM图像特征
        """
        input_size = images.shape[2:]

        sam_features = self.sam_processor(sam_embedding)
        x_image = self.image_processor(images)
        x_mask = self.mask_processor(teeth_mask)

        # 编码阶段 + 融合
        e1_image = self.image_encoder1(x_image)
        e1_mask = self.mask_encoder1(x_mask)
        e1_fusion = self.fusion1(e1_image, e1_mask, sam_features['level1'])

        x_image = F.max_pool2d(e1_image, 2)
        x_mask = F.max_pool2d(e1_mask, 2)

        e2_image = self.image_encoder2(x_image)
        e2_mask = self.mask_encoder2(x_mask)
        e2_fusion = self.fusion2(e2_image, e2_mask, sam_features['level2'])

        x_image = F.max_pool2d(e2_image, 2)
        x_mask = F.max_pool2d(e2_mask, 2)

        e3_image = self.image_encoder3(x_image)
        e3_mask = self.mask_encoder3(x_mask)
        e3_fusion = self.fusion3(e3_image, e3_mask, sam_features['level3'])

        x_image = F.max_pool2d(e3_image, 2)
        x_mask = F.max_pool2d(e3_mask, 2)

        e4_image = self.image_encoder4(x_image)
        e4_mask = self.mask_encoder4(x_mask)
        e4_fusion = self.fusion4(e4_image, e4_mask, sam_features['level4'])

        # 瓶颈
        x = F.max_pool2d(e4_fusion, 2)
        x = self.bottleneck(x)

        # 解码
        x = self.decoder4(x, e4_fusion)
        x = self.decoder3(x, e3_fusion)
        x = self.decoder2(x, e2_fusion)
        x = self.decoder1(x, e1_fusion)

        # 输出到原图尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        logits = self.final_conv(x)
        return logits 