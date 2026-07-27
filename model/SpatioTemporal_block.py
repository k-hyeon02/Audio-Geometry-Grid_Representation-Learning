import torch
from torch.nn.modules.activation import MultiheadAttention
from .util import channelwise_softmax_aggregation, LayerNorm

class Dualpath_block(torch.nn.Module):
    def __init__(self, num_heads, feature_size, rnn_layers=2):
        super(Dualpath_block, self).__init__()

        self.mhsa = MultiheadAttention(num_heads=num_heads, embed_dim=feature_size)
        self.rnn = torch.nn.GRU(input_size=2*feature_size, hidden_size=feature_size, bidirectional=False, batch_first=False, num_layers=rnn_layers)

        self.linear = torch.nn.Linear(feature_size, feature_size)

        self.norm_mhsa = LayerNorm(feature_size)
        self.norm_rnn = LayerNorm(feature_size)

        self.activation = torch.nn.ELU()

    def mhsa_forward(self, x, MPE):
        B, P, F, T = x.shape  # (B, P, F, T)
        x_steered = x + MPE

        x_steered = x_steered.permute(0, 3, 1, 2).contiguous().view(B*T, P, F)
        x_steered = x_steered.permute(1, 0, 2) # (P, B*T, F)
        x_steered, _ = self.mhsa(x_steered, x_steered, x_steered)  # Linear layer 마지막에 포함되어 있음

        x_steered = x_steered.permute(1, 0, 2)  # (B*T, P, F)
        x_steered = x_steered.contiguous().view(B, T, P, F).permute(0, 2, 3, 1).contiguous().view(B*P, F, T)

        out = self.norm_mhsa(x_steered)
        out = out.view(B, P, F, T)  # (B, P, F, T)

        return out

    def rnn_forward(self, x):
        B, _P, F, T = x.shape  # (B, P, F, T); P 축은 aggregation에서 사라져 미사용
        out = x.clone()
        out=channelwise_softmax_aggregation(out, std=True)

        out = out.permute(2, 0, 1)  # (T, B, 2F)
        out=self.rnn(out)[0]  # (T, B, F)
        out = out.permute(1, 2, 0)  # (B, F, T)

        out=self.activation(out)
        out=self.linear(out.transpose(-2, -1)) # (B, T, F)
        out = self.norm_rnn(out.transpose(-2, -1))  # (B, F, T)

        out = out.view(B, 1, F, T)  # (B, 1, F, T)
        return out
    
    def forward(self, x, MPE):

        out = self.mhsa_forward(x, MPE)  # Apply multi-head self-attention        
        x = x + out

        out = self.rnn_forward(x)  # Apply RNN
        x = x + out
        return x
    
class SpatioTemporal_block(torch.nn.Module):
    def __init__(self, n_heads, num_blocks, feature_size, rnn_layers):
        super(SpatioTemporal_block, self).__init__()     

        self.dualpath_block_list=torch.nn.ModuleList()  

        for i in range(num_blocks):    
            self.dualpath_block_list.append(
                Dualpath_block(
                    num_heads=n_heads,
                    feature_size=feature_size,
                    rnn_layers=rnn_layers))
            
    def forward(self, x, MPE):
        _B, _P, _F, T=x.shape  # (B, P, F, T); T만 MPE 시간축 확장에 사용
        MPE=MPE.unsqueeze(-1)
        MPE=torch.repeat_interleave(MPE, T, dim=-1)        
        for i in range(len(self.dualpath_block_list)):
            x = self.dualpath_block_list[i](x, MPE) 

        return x 