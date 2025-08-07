import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class ConvBNReLU(nn.Module):
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
            bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DSPWrapper(nn.Module):
    def __init__(
        self, 
        in_channels, 
        deformable_groups=1, 
        scale_factor=1.0,
        offset_channels=None,  
        conv_channels=None,   
        kernel_size=3,       
        use_residual=True,     
        use_bn=True,          
        use_relu=True   
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.deformable_groups = int(deformable_groups)
        self.scale_factor = float(scale_factor)
        self.kernel_size = int(kernel_size)
        self.use_residual = bool(use_residual)
        self.use_bn = bool(use_bn)
        self.use_relu = bool(use_relu)
        
        if offset_channels is None:
            self.offset_channels = self.in_channels // 4
        else:
            self.offset_channels = int(offset_channels)
        
        if conv_channels is None:
            self.conv_channels = self.in_channels
        else:
            self.conv_channels = int(conv_channels)

        self.offset_net = nn.Sequential(
            ConvBNReLU(
                self.in_channels, 
                self.offset_channels, 
                kernel_size=1, 
                use_bn=self.use_bn, 
                use_relu=self.use_relu
            ),
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
            nn.Conv2d(self.offset_channels // 2, 2, kernel_size=1)
        )
        self.conv = nn.Sequential(
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
        for p in self.offset_net[-1].parameters():
            nn.init.zeros_(p)
            
    def forward(self, x):
        B, H, W, C = x.shape

        x_in = x.permute(0, 3, 1, 2).contiguous()
        identity = x_in if self.use_residual else None
        offset = self.offset_net(x_in)
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )

        grid = torch.stack([grid_w, grid_h], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        offset_permuted = offset.permute(0, 2, 3, 1).contiguous()
        offset_scaled = offset_permuted * self.scale_factor / torch.tensor([W/2., H/2.], device=offset.device)
        grid_with_offset = grid + offset_scaled
        x_sampled = F.grid_sample(
            x_in, grid_with_offset, mode='bilinear', padding_mode='zeros', align_corners=True
        )

        x_conv = self.conv(x_sampled)
        if self.use_residual and identity is not None:
            x_out_feat = x_conv + identity
        else:
            x_out_feat = x_conv
        x_out = x_out_feat.permute(0, 2, 3, 1).contiguous()

        return x_out