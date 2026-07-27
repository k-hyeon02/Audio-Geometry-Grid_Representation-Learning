import torch
from .util import ConvBlock, ResidualBlock

class Channel_invariant_feature_extractor(torch.nn.Module):
    def __init__(self, init_feature, feature, kernel_size = 3, padding = 1, stride = 1, dilation_rate = 2, num_blocks = 4):   
        super(Channel_invariant_feature_extractor, self).__init__() 

        self.init_BN=torch.nn.BatchNorm1d(init_feature)     
        self.init_ConvBlock=ConvBlock(init_feature+8, feature, kernel_size, stride, padding, norm = 'BN')

        self.ResidualConvBlocks=torch.nn.ModuleList()
        for i in range(num_blocks):
            dilation=dilation_rate**i 
            self.ResidualConvBlocks.append(ResidualBlock(feature, 
                                                        kernel_size,
                                                        dilation, 
                                                        dilation,
                                                        norm='BN'))   
             
    def forward(self, x, MPE, mic_coord_cart, mic_coord_dist_sin_cos):
        # x: (B, P, 2K, T) = (B, C-1, 514, T)
        # FK: 입력 2K(=514) freq-concat 차원 (ConvBlock을 거쳐 feature F=128이 됨)
        B, P, FK, T=x.shape
        MPE = MPE.view(B*P, -1, 1).repeat_interleave(T, dim=-1)  # (B*P, F, T), F=128

        x=x.view(B*P, FK, T)  # (B*P, 2K, T)
        x=self.init_BN(x)

        mic_coord = torch.cat([mic_coord_cart, mic_coord_dist_sin_cos], dim=-1)  # (B, P, 8)
        mic_coord= mic_coord.view(B*P, -1, 1).repeat_interleave(T, dim=2)  # (B*P, 8, T)
        x=torch.cat([x, mic_coord], dim=1   )  # (B*P, 2K+8, T) = (B*P, 522, T)
        x=self.init_ConvBlock(x)  # (B*P, F, T) = (B*P, 128, T)

        for i in range(len(self.ResidualConvBlocks)):
            x= x+MPE
            x=self.ResidualConvBlocks[i](x)  # (B*P, F, T)

        x=x.view(B, P, x.shape[1], T)  # (B, P, F, T) = (B, C-1, 128, T)

        return x