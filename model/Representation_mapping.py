import torch
from .util import channelwise_softmax_aggregation, ResidualBlock

class Representation_mapping_block(torch.nn.Module):
    def __init__(self, feature, representation_size, num_blocks, kernel_size=3, dilation_rate=2,):
        super(Representation_mapping_block, self).__init__()

        self.representation_size = representation_size

        self.ResidualConvBlocks=torch.nn.ModuleList()
        self.mapping_layers=torch.nn.ModuleList()

        for i in range(num_blocks):
            dilation=dilation_rate**i 

            self.ResidualConvBlocks.append(ResidualBlock(2*feature,
                                                         kernel_size,
                                                         dilation,
                                                         dilation,
                                                         norm='LN'))

            self.mapping_layers.append(torch.nn.Linear(2*feature,representation_size))

    def forward(self, x):
        # (B, P, F, T) → (B, 2F, T)
        x = channelwise_softmax_aggregation(x, std=True)

        outputs=[]

        for i in range(len(self.ResidualConvBlocks)):
            x = self.ResidualConvBlocks[i](x)
            # x: (B, 2F, T) → (B, T, 2F)로 바꿔서 입력
            # Linear 후 (B, T, R) → (B, R, T)
            out = self.mapping_layers[i](x.transpose(1, 2)).transpose(1, 2)
            outputs.append(out)

        # 3개의 depth 결과를 새 축 dim=1에 쌓아 (B, DS, R, T)
        outputs=torch.stack(outputs, dim=1)

        return outputs  # (B, DS, R, T)
