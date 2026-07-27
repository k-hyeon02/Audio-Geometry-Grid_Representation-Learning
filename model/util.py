import torch
from torch import nn
import math

def get_cos_distance(v1, v2, return_type='radian'):
    """
    코사인 유사도를 이용해 계산한 두 벡터 사이의 각도(geodesic angle) 반환하는 함수
    """
    v1_size=torch.linalg.norm(v1, dim=-1, keepdim=True)
    v1_size=torch.where(v1_size==0.0, 1e-12, v1_size)
	
    v1=v1/v1_size
    v2=v2/torch.linalg.norm(v2, dim=-1, keepdim=True)

    dotted=torch.sum(v1*v2, dim=-1)
    angle=torch.arccos(torch.clip(dotted, -1.0, 1.0))

    if return_type=='degree':
        return torch.rad2deg(angle) % 360
    else:
        return angle


def fibonacci_sphere(num_points):
        
    indices = torch.arange(num_points, dtype=torch.float32)
    z = 1 - (indices / (num_points - 1)) * 2
       
    radius = torch.sqrt(1 - z * z)
    golden_angle = math.pi * (3 - math.sqrt(5))  # ≈ 2.39996
    theta = golden_angle * indices

    x = torch.cos(theta) * radius
    y = torch.sin(theta) * radius

    points = torch.stack((x, y, z), dim=1)

    return points

def target_spatial_spectrum(target_spherical_position, DOA_candidates, vad_framed, gammas, framing_func):
    """
    격자점이 화자의 실제 방향과 정확히 일치하면 값이 1,
    멀어질수록 gamma(빔폭)에 따라 정해진 속도로 부드럽게 0에 가까워지는 연속값 가우시안형(vMF) 히트맵을 만들고,
    5번 단계에서 VAD와 곱해지고 화자 간 max를 취해 최종 ground-truth spatial spectrum 생성

    len(gammas)는 반환값의 DS(depth-stage) 축 크기가 되며, 이는 loss.py에서 element-wise로 비교되는
    model.main.AGG_RL.forward()의 x_out(GridNet num_blocks에서 나온 DS 축)과 반드시 일치해야 함
    """
    # 1. 구면좌표 → 직교좌표 변환 target_xyz_position: (B, Q, 3, T)
    x,y,z=sph2cart(target_spherical_position[..., 0, :], target_spherical_position[...,1, :], target_spherical_position[...,2, :], is_degree=True)
    target_xyz_position=torch.stack((x,y,z), dim=-1).transpose(2,3) 
    target_xyz_poisition_framed=framing_func(target_xyz_position) 
    target_xyz_poisition_framed=target_xyz_poisition_framed.unsqueeze(2) 
    target_xyz_poisition_framed=target_xyz_poisition_framed.transpose(-1, -2)  # (B, Q, 1, T, 3)

    # 2. DOA candidate 격자 준비
    # reshape: (D=2048, 3) -> (1, 1, D, 3, 1)
    DOA_candidates=DOA_candidates.view(1, 1, DOA_candidates.shape[0], 3, 1).to(target_xyz_poisition_framed.device)
    # (D, 3)을 브로드캐스팅 가능하게 (1, 1, D, 1, 3)으로 reshape
    DOA_candidates=DOA_candidates.transpose(-1, -2)

    # 3. 각도 거리 계산
    # 각 화자의 실제 위치와 D개 격자점 사이 각도 차이 계산 → candidate_distance: (B, Q, D, T)
    candidate_distance=get_cos_distance(target_xyz_poisition_framed, DOA_candidates, return_type='radian')

    # 4. von Mises-Fisher 커널 적용
    gammas = torch.tensor(gammas, dtype=torch.float32, device=candidate_distance.device).view(1, -1, 1, 1)  
    gammas = torch.deg2rad(gammas)  # (1, DS, 1, 1)

    # f(theta) = exp(kappa(cos(theta) - 1))
    # 각도가 gamma일 때 값이 half-power가 되도록 하는 kappa를 역산
    kappa = math.log(0.5**0.5) / (torch.cos(gammas) - 1).unsqueeze(2)  # (1, DS, 1, 1, 1)
    candidate_distance = candidate_distance.unsqueeze(1)  # (B, 1, Q, D, T)
    candidate_distance = torch.exp(kappa * (torch.cos(candidate_distance) - 1))  # (B, DS, Q, D, T)

    # 5. VAD(발화 여부) 마스킹 + 화자 간 max 병합
    # 여러 화자(Q명)가 있을 때, 격자점별로 가장 강한 화자의 활성값을 취함 (multi-speaker spectrum 합성)
    vad_framed=vad_framed.view(vad_framed.shape[0], 1, vad_framed.shape[1],  1, -1)
    target = vad_framed * candidate_distance  
    target = torch.max(target, dim=2).values  

    return target  # (B, DS, D, T)

def channelwise_softmax_aggregation(x: torch.Tensor, std=True):
    """
    (B, P, F, T)
    B: batch
    P: 기준 마이크 대비 비기준 마이크 수 (= C - 1, channel-pair 축)
    F: feature 차원 (128)
    T: 시간 frame
    """
    out_softmax = x.softmax(dim=1) # (B, P, F, T) → (B, F, T)
    out = out_softmax * x  # (B, P, F, T)

    out_sum = out.sum(dim=1, keepdim=False)  # (B, F, T)

    if std:
        out_std = out.std(dim=1, keepdim=False)  # (B, F, T)
        out = torch.cat([out_sum, out_std], dim=1)  # (B, 2F, T)
    else:
        out= out_sum

    return out

def sph2cart(azimuth, elevation, distance, is_degree=True):
    if is_degree:
        azimuth=torch.deg2rad(azimuth)
        elevation=torch.deg2rad(elevation)
    
    x=distance*torch.sin(elevation)*torch.cos(azimuth)
    y=distance*torch.sin(elevation)*torch.sin(azimuth)
    z=distance*torch.cos(elevation)
    return x, y, z

def cart2sph(x, y, z, is_degree=True):
    
    azimuth=torch.atan2(y, x)
    elevation=torch.pi/2-torch.atan2(z, torch.sqrt(x**2+y**2))
    distance=torch.sqrt(x**2+y**2+z**2)

    if is_degree:
        azimuth=torch.rad2deg(azimuth)
        elevation=torch.rad2deg(elevation)

    return azimuth, elevation, distance   

# this is from Asteroid: https://github.com/asteroid-team/asteroid

class LayerNorm(nn.Module):

    def __init__(self, feature_size):
        super(LayerNorm, self).__init__()
        self.feature_size = feature_size
        self.gamma = nn.Parameter(torch.ones(feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(feature_size), requires_grad=True)

    def forward(self, x, EPS: float = 1e-8):        

        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())
    
    def apply_gain_and_bias(self, normed_x):
 
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class ResidualBlock(nn.Module):

    def __init__(self,
                 feature,
                 kernel,
                 padding,
                 dilation=1,
                 norm='BN'):
        
        super(ResidualBlock, self).__init__()
        
        self.padding = (kernel - 1) * dilation
 
        self.pw_conv1d = nn.Conv1d(feature, feature, 1)
        self.dw_conv1d = nn.Conv1d(feature,
                                 feature,
                                 kernel,
                                 dilation=dilation,
                                 groups=feature,
                                 padding=self.padding)

        if norm=='BN':
            self.pw_norm=nn.BatchNorm1d(feature)
            self.dw_norm=nn.BatchNorm1d(feature)        
        elif norm=='LN':
            self.pw_norm=LayerNorm(feature)
            self.dw_norm=LayerNorm(feature)
        else:
            raise ValueError('Not exist normalization method')

        self.activation = nn.ELU()
 

    def forward(self, input):

        output=self.pw_conv1d(input)
        output=self.activation(output)
        output=self.pw_norm(output)

        output=self.dw_conv1d(output)  
        output=output[...,:-self.padding]  
        output=self.activation(output)
        output=self.dw_norm(output)

        output=output+input    
        
        return output

class ConvBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 norm='BN'):
        
        super(ConvBlock, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1d=torch.nn.Conv1d(in_features,
                                    out_features,
                                    kernel_size,
                                    stride,
                                    padding=self.padding, 
                                    dilation=dilation,
                                    groups=groups)
        
        if norm=='BN':
            self.norm=torch.nn.BatchNorm1d(out_features)
        elif norm=='LN':
            self.norm=LayerNorm(out_features)
        else:
            raise ValueError('Not exist normalization method')
        
        self.activation=torch.nn.ELU()

    def forward(self, x):
        
        x=self.conv1d(x)
        x = x[..., :-self.padding]
        x=self.activation(x)
        x=self.norm(x)
        return x
