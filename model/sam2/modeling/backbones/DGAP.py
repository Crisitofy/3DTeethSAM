import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class ConvBNReLU(nn.Module):
    """基础卷积块：Conv + BN + ReLU"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3,         
        stride=1, 
        groups=1,
        use_bn=True,
        use_relu=True
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not use_bn  # 如果使用BN，则不需要偏置
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DSPWrapper(nn.Module):
    """DSP包装器，用于在backbone中集成可变形采样功能"""
    def __init__(
        self, 
        in_channels, 
        deformable_groups=1, 
        scale_factor=1.0,
        offset_channels=None,  # 偏移网络的通道数
        conv_channels=None,    # 卷积层的通道数
        kernel_size=3,         # 卷积核大小
        use_residual=True,     # 是否使用残差连接
        use_bn=True,          # 是否使用批归一化
        use_relu=True         # 是否使用ReLU激活
    ):
        super().__init__()
        
        # 确保参数类型正确
        self.in_channels = int(in_channels)
        self.deformable_groups = int(deformable_groups)
        self.scale_factor = float(scale_factor)
        self.kernel_size = int(kernel_size)
        self.use_residual = bool(use_residual)
        self.use_bn = bool(use_bn)
        self.use_relu = bool(use_relu)
        
        # 设置偏移网络的通道数
        if offset_channels is None:
            self.offset_channels = self.in_channels // 4  # 默认值：输入通道数的1/4
        else:
            self.offset_channels = int(offset_channels)
        
        # 设置卷积层的通道数
        if conv_channels is None:
            self.conv_channels = self.in_channels  # 默认值：与输入通道数相同
        else:
            self.conv_channels = int(conv_channels)
            
        # 打印初始化信息
        print(f"DSP配置")
        # print(f"- 输入通道数: {self.in_channels}")
        # print(f"- 可变形组数: {self.deformable_groups}")
        # print(f"- 缩放因子: {self.scale_factor}")
        # print(f"- 偏移通道数: {self.offset_channels}")
        # print(f"- 卷积通道数: {self.conv_channels}")
        # print(f"- 卷积核大小: {self.kernel_size}")
        # print(f"- 使用残差: {self.use_residual}")
        # print(f"- 使用BN: {self.use_bn}")
        # print(f"- 使用ReLU: {self.use_relu}")
            
        # 偏移预测网络
        self.offset_net = nn.Sequential(
            # 第一个卷积块：标准卷积，减少通道数
            ConvBNReLU(
                self.in_channels, 
                self.offset_channels, 
                kernel_size=1, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            ),
            # 深度可分离卷积（没有实现）
            ConvBNReLU(
                self.offset_channels, 
                self.offset_channels, 
                kernel_size=self.kernel_size, 
                groups=self.offset_channels, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            ),
            ConvBNReLU(
                self.offset_channels, 
                self.offset_channels // 2, 
                kernel_size=1, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            ),
            # 输出偏移
            nn.Conv2d(self.offset_channels // 2, 2, kernel_size=1)
        )
        
        # 主要处理分支
        self.conv = nn.Sequential(
            # 深度可分离卷积
            ConvBNReLU(
                self.in_channels, 
                self.conv_channels, 
                kernel_size=self.kernel_size, 
                groups=self.deformable_groups, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            ),
            ConvBNReLU(
                self.conv_channels, 
                self.in_channels, 
                kernel_size=1, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            )
        )
        
        # 初始化偏移预测层为0
        for p in self.offset_net[-1].parameters():
            nn.init.zeros_(p)
            
    def forward(self, x, visualize=False):
        # 输入x的形状为 [B, H, W, C]
        B, H, W, C = x.shape
        
        # 验证输入通道数是否正确
        assert C == self.in_channels, f"输入通道数 {C} 与初始化通道数 {self.in_channels} 不匹配"
        
        # 转换为 [B, C, H, W] 格式
        x_in = x.permute(0, 3, 1, 2).contiguous()
        identity = x_in if self.use_residual else None
        
        # 预测偏移
        offset = self.offset_net(x_in)  # [B, 2, H, W]
        
        # 生成基础网格
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_w, grid_h], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # 将偏移转换为采样网格的偏移
        offset_permuted = offset.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2]
        offset_scaled = offset_permuted * self.scale_factor / torch.tensor([W/2., H/2.], device=offset.device)
        
        # 应用偏移到采样网格
        grid_with_offset = grid + offset_scaled
        
        # 使用网格采样
        x_sampled = F.grid_sample(
            x_in, grid_with_offset, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # 应用卷积
        x_conv = self.conv(x_sampled)
        
        # 应用残差连接
        if self.use_residual and identity is not None:
            x_out_feat = x_conv + identity
        else:
            x_out_feat = x_conv
        
        # 转换回 [B, H, W, C] 格式
        x_out = x_out_feat.permute(0, 2, 3, 1).contiguous()

        vis_data = None
        if visualize:
            vis_data = {
                'input_feat': x_in.detach().cpu(),
                'offset': offset.detach().cpu(),
                'output_feat': x_out_feat.detach().cpu(),
                'original_grid': grid.detach().cpu(),
                'deformed_grid': grid_with_offset.detach().cpu()
            }
        
        return x_out, vis_data 