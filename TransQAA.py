import copy
import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from utils import parse_args

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def trunc_normal_(tensor, std):
    nn.init.trunc_normal_(tensor, std=std)

class AgentAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, window_size=(64,64), num_heads=8,
                 qkv_bias=True, attn_drop=0., proj_drop=0., shift_size=0, agent_num=49, width=False):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.dim = out_channels
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=(3, 3), padding=1, groups=self.dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.width = width

    def forward(self, x, window_size=None, mask=None):
        if window_size is not None:
            self.window_size = window_size
            self.ah_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, window_size[0], 1))
            self.aw_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, 1, window_size[1]))
            self.ha_bias = nn.Parameter(torch.zeros(1, self.num_heads, window_size[0], 1, self.agent_num))
            self.wa_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, window_size[1], self.agent_num))

        x = self.input_proj(x)  

        if self.width:
            x = x.permute(0, 2, 3, 1)
            b, h, w, c = x.shape
        else:
            x = x.permute(0, 3, 2, 1)
            b, w, h, c = x.shape
        n = h * w

        x = x.reshape(h * b, w, c)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        qt = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2))
        qt = qt.reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        qt = qt.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = F.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        qt_attn = self.softmax((qt * self.scale) @ k.transpose(-2, -1) + position_bias)
        qt_attn = self.attn_drop(qt_attn)
        agent_v = qt_attn @ v

        bias1 = F.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        bias1 = bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        bias = bias1 + bias2
        q_attn = self.softmax((q * self.scale) @ qt.transpose(-2, -1) + bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(b, h, w, c)
        x = x.permute(0, 3, 1, 2)  
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=64):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)


        self.hight_block = AgentAttention(in_channels=width, out_channels=width, num_heads=8)
        self.width_block = AgentAttention(in_channels=width, out_channels=width, num_heads=8, width=True)

        self.conv_up = conv1x1(width, inplanes)
        self.bn2 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.ff = PositionwiseFeedForward(kernel_size, 256, 0.1)

    def forward(self, x):
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        identity = out
        _,_,H,W = out.shape
        out = self.hight_block(out, window_size=(H, W))

        out += identity
        identity = out
        _, _, H, W = out.shape
        out = self.width_block(out, window_size=(H, W))

        out += identity
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)
        identity = out
        out = self.ff(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(2)],
                         requires_grad=False)
        return self.dropout(x)


class TransAxials(nn.Module):
    def __init__(self, d_model=256, d_ff=256, dropout=0.1):
        super(TransAxials, self).__init__()
        self.bn = nn.LayerNorm([d_model, d_model])
        self.bn2 = nn.LayerNorm([d_model // 2, d_model // 2])
        self.bn3 = nn.LayerNorm([d_model // 4, d_model // 4])
        self.bn4 = nn.LayerNorm([d_model // 8, d_model // 8])
        self.position = PositionalEncoding(d_model, dropout)
        self.axial_layer = AxialBlock(3, 32, kernel_size=256)
        self.axial_layer2 = AxialBlock(6, 32, kernel_size=128)
        self.axial_layer3 = AxialBlock(12, 32, kernel_size=64)
        self.axial_layer4 = AxialBlock(24, 32, kernel_size=32)
        # self.classifty = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),  
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),  
        #
        #     nn.MaxPool2d(2, 2),
        #     nn.Dropout(0.5),
        #     nn.Linear(64 * 16 * 16, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 5)
        # )

    def forward(self, x):
        x = self.position(x)
        
        x = self.bn(x)
        x = self.axial_layer(x)
        down = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        
        x = down(x)
        x = self.bn2(x)
        x = self.axial_layer2(x)
        down = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        
        x = down(x)
        x = self.bn3(x)
        x = self.axial_layer3(x)
        down = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        
        x = down(x)
        x = self.bn4(x)
        x = self.axial_layer4(x)
        
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24*32*32, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        B, C,H,W = x.shape
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train(self, features, labels, criterion, optimizer_class, num):
        args = parse_args()
        for epoch in range(args.class_epochs):
            outputs = self(features)
            
            loss_train = criterion(outputs, labels)
            
            if epoch ==0 or epoch == 49:
                print(f"Classifier loss: Epoch {epoch + 1}, Class Loss: {loss_train.item():.4f}")
                
            optimizer_class.zero_grad()
            loss_train.backward()
            optimizer_class.step()

    def eval(self):
        self.training = False
        for module in self.children():
            module.training = False

        return self

if __name__ == '__main__':
    x = torch.randn(24, 3, 256, 256)
    ta = TransAxials()
    out = ta(x)
    print(out.shape)  # torch.Size([24, 24, 32, 32])



