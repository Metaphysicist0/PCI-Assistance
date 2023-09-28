# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# 定义一个Transformer模块，包含一个多头自注意力层和一个前馈层，以及残差连接和层归一化
class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 计算每个头的维度
        assert out_channels % num_heads == 0
        self.head_dim = out_channels // num_heads

        # 定义多头自注意力层的参数
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.attn_combine = nn.Linear(out_channels, out_channels)

        # 定义前馈层的参数
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels)
        )

        # 定义残差连接和层归一化
        self.res1 = nn.Identity()
        self.res2 = nn.Identity()
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: [batch_size, height * width, in_channels]

        # 计算查询、键和值
        q = self.q_proj(x)  # [batch_size, height * width, out_channels]
        k = self.k_proj(x)  # [batch_size, height * width, out_channels]
        v = self.v_proj(x)  # [batch_size, height * width, out_channels]

        # 重塑为多头形式
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads,
                      self.head_dim)  # [batch_size, height * width, num_heads, head_dim]
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads,
                      self.head_dim)  # [batch_size, height * width, num_heads, head_dim]
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads,
                      self.head_dim)  # [batch_size, height * width, num_heads, head_dim]

        # 转置为[batch_size, num_heads, height * width, head_dim]
        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        # 计算注意力权重
        attn_scores = torch.matmul(q / self.attn_scale,
                                   k.transpose(-1, -2))  # [batch_size,num_heads,height*width,height*width]

        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size,num_heads,height*width,height*width]
        attn_weights = self.attn_dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch_size,num_heads,height*width,out_dim]
        # 转置回[batch_size,height*width,num_heads,out_dim]
        attn_output = attn_output.permute(0, 2, 1, 3)
        # 合并多头并通过线性层
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], self.out_channels)
        # [batch_size,height*width,out_channels]
        attn_output = self.attn_combine(attn_output)  # [batch_size,height*width,out_channels]
        # 添加残差连接和层归一化
        attn_output = self.norm1(self.res1(x) + attn_output)  # [batch_size,height*width,out_channels]
        # 通过前馈层
        ffn_output = self.ffn(attn_output) # [batch_size,height*width,out_channels]
        # 添加残差连接和层归一化
        ffn_output = self.norm2(self.res2(attn_output) + ffn_output) # [batch_size,height*width,out_channels]

        return ffn_output


# 定义一个ResBlock模块，包含一个卷积层、一个Transformer模块和一个下采样层
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 定义Transformer模块
        self.transformer = Transformer(out_channels, out_channels, num_heads, dropout)
        # 定义下采样层
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        # x: [batch_size,in_channels,height,width]

        # 通过卷积层
        x = self.conv(x)  # [batch_size,out_channels,height,width]
        # 重塑为[batch_size,height*width,out_channels]
        x = x.reshape(x.shape[0], -1, x.shape[1]) # [batch_size,height*width,out_channels]
        # 通过Transformer模块
        x = self.transformer(x)  # [batch_size,height*width,out_channels]
        # 重塑回[batch_size,out_channels,height,width]
        x = x.reshape(x.shape[0], x.shape[-1], int(torch.sqrt(x.shape[1])), int(torch.sqrt(x.shape[1]))) # [batch_size,out_channels,height,width]
        # 通过下采样层
        x = self.downsample(x)  # [batch_size,out_channels,height/2,width/2]

        return x


# 定义一个UpBlock模块，包含一个上采样层、一个卷积层、一个Transformer模块和一个跳跃连接
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.1):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义上采样层
        self.upsample = nn.Upsample(scale_factor=2)
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        # 定义Transformer模块
        self.transformer = Transformer(out_channels ,out_channels ,num_heads ,dropout)
        # 定义跳跃连接
        self.skip_connect = nn.Identity()

        def forward(self ,x ,skip):
         # x: [batch_size,in_channels,height,width]
         # skip: [batch_size,out_channels,height*2,width*2]

             # 通过上采样层
            x = self.upsample(x) # [batch_size,in_channels,height*2,width*2]
            # 沿通道维度拼接跳跃连接的特征图
            x = torch.cat([x ,skip] ,dim=1) # [batch_size,in_channels+out_channels,height*2,width*2]
            # 通过卷积层
            x = self.conv(x) # [batch_size,out_channels,height*2,width*2]
            # 重塑为[batch_size,height*2*width*2,out_channels]
            x = x.reshape(x.shape[0], -1, x.shape[1])  # [batch_size,height*2*width*2,out_channels]
            # 通过Transformer模块
            x = self.transformer(x)  # [batch_size,height*2*width*2,out_channels]
            # 重塑回[batch_size,out_channels,height*2,width*2]
            x = x.reshape(x.shape[0], x.shape[-1], int(torch.sqrt(x.shape[1] / 4)),
                      int(torch.sqrt(x.shape[1] / 4)))  # [batch_size,out_channels,height*2,width*2]

            return x


# 定义一个输出层，包含一个卷积层和一个激活函数
class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 定义激活函数
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: [batch_size,in_channels,height,width]
        # 通过卷积层
        x = self.conv(x) # [batch_size,out_channels,height,width]
        # 通过激活函数
        x = self.activation(x) # [batch_size,out_channels,height,width]

        return x


# 定义一个TransformerResUNet类，用于初始化和前向传播
class TransResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4, dropout=0.1):
        super(TransResUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义编码器部分的四个ResBlock
        self.encoder1 = ResBlock(in_channels, 64, num_heads, dropout)
        self.encoder2 = ResBlock(64, 128, num_heads, dropout)
        self.encoder3 = ResBlock(128, 256, num_heads, dropout)
        self.encoder4 = ResBlock(256, 512, num_heads, dropout)

        # 定义解码器部分的四个UpBlock
        self.decoder1 = UpBlock(512, 256, num_heads, dropout)
        self.decoder2 = UpBlock(256, 128, num_heads, dropout)
        self.decoder3 = UpBlock(128, 64, num_heads, dropout)
        self.decoder4 = UpBlock(64, 64, num_heads, dropout)

        # 定义输出层
        self.output_layer = OutputLayer(64, out_channels)

    def forward(self, x):
        # x: [batch_size,in_channels,height,width]

        # 通过编码器部分的四个ResBlock
        e1 = self.encoder1(x)  # [batch_size ,64 ,height/2 ,width/2]
        e2 = self.encoder2(e1)  # [batch_size ,128 ,height/4 ,width/4]
        e3 = self.encoder3(e2)  # [batch_size ,256 ,height/8 ,width/8]
        e4 = self.encoder4(e3)  # [batch_size ,512 ,height/16 ,width/16]

        # 通过解码器部分的四个UpBlock
        d1 = self.decoder1(e4, e3)  # [batch_size ,256 ,height/8 ,width/8]
        d2 = self.decoder2(d1, e2)  # [batch_size ,128 ,height/4 ,width/4]
        d3 = self.decoder3(d2, e1)  # [batch_size ,64 ,height/2 ,width/2]
        d4 = self.decoder4(d3)  # [batch_size ,64 ,height,width]

        # 通过输出层
        output = self.output_layer(d4)  # [batch_size,out_channels,height,width]

        return output


if __name__ == "__main__":
    net = TransResUNet()
    print(net)