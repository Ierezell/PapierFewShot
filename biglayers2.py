from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch
from utils import adaIN

"""

https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
Suplementary material pour l'archi de base [19] a modifier ensuite avec
l'attention

TIRÉ DE L'ARTICLE :

We base our generator network G(yi(t), eˆi; φ, P) on the image-to-image
translation architecture proposed by Johnson et. al. [19],
but replace downsampling and upsampling
layers with residual blocks similarly to [2] (with batch normalization [15]
replaced by instance normalization [36]).
The person-specific parameters ψˆ
i serve as the affine coefficients of instance normalization layers,
following the adaptive instance normalization technique proposed in [14],
though we still use regular (non-adaptive) instance normalization layers
in the downsampling blocks that encode landmark images yi(t).



For the embedder E(xi(s), yi(s); φ) and the convolutional part of the
discriminator V (xi(t), yi(t); θ), we use
similar networks, which consist of residual downsampling
blocks (same as the ones used in the generator, but without
normalization layers).



The discriminator network, compared to the embedder, has an additional
residual block at
the end, which operates at 4×4 spatial resolution. To obtain
the vectorized outputs in both networks, we perform global
sum pooling over spatial dimensions followed by ReLU

We use spectral normalization [33] for all convolutional
and fully connected layers in all the networks

We also use
self-attention blocks, following [2] and [42].
They are inserted at 32×32 spatial resolution in all downsampling parts
of the networks and at 64×64 resolution in the upsampling
part of the generator.


[2]  https://arxiv.org/pdf/1809.11096.pdf  Page19 pour les Resblock down/up
[14] https://arxiv.org/pdf/1703.06868.pdf
[15] https://arxiv.org/pdf/1502.03167.pdf
[19] https://arxiv.org/pdf/1603.08155.pdf
[33] https://arxiv.org/pdf/1802.05957.pdf
[36] https://arxiv.org/pdf/1607.08022.pdf  Instance norms
"""


class BigResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BigResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temp_channels = max(1, in_channels // 4)
        self.params = in_channels + 3*self.temp_channels
        self.adaDim = spectral_norm(nn.Conv2d(in_channels,  out_channels,
                                              kernel_size=1, padding=0,
                                              bias=False))
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, self.temp_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.conv2 = spectral_norm(nn.Conv2d(self.temp_channels,
                                             self.temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv3 = spectral_norm(nn.Conv2d(self.temp_channels,
                                             self.temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv4 = spectral_norm(nn.Conv2d(self.temp_channels, out_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.relu = nn.SELU()

    def forward(self, x, w):
        residual = self.relu(self.adaDim(x))

        out = self.relu(x)
        out = self.conv1(out)
        out = adaIN(out, mean_style=w, std_style=w)

        out = self.relu(out)
        out = self.conv2(out)
        out = adaIN(out, mean_style=w, std_style=w)

        out = self.relu(out)
        out = self.conv3(out)
        out = adaIN(out, mean_style=w, std_style=w)

        out = self.relu(out)
        out = self.conv4(out)
        out = adaIN(out, mean_style=w, std_style=w)
        out += residual
        return out


class BigResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BigResidualBlockDown, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        temp_channels = max(1, in_channels // 4)
        # if out_channels - in_channels > 0:
        #     self.adaDim = spectral_norm(nn.Conv2d(in_channels,
        #                                           out_channels-in_channels,
        #                                           kernel_size=1, padding=0,
        #                                           bias=False))
        self.adaDim = spectral_norm(nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1, padding=0,
                                              bias=False))
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, temp_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.conv2 = spectral_norm(nn.Conv2d(temp_channels, temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv3 = spectral_norm(nn.Conv2d(temp_channels, temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv4 = spectral_norm(nn.Conv2d(temp_channels, out_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.avgPool = nn.AvgPool2d(kernel_size=2)

        self.relu = nn.SELU()

    def forward(self, x):
        residual = x

        residual = self.avgPool(residual)

        residual = self.adaDim(residual)

        out = self.relu(x)
        out = self.conv1(out)

        out = self.relu(out)
        out = self.conv2(out)

        out = self.relu(out)
        out = self.conv3(out)

        out = self.relu(out)
        out = self.avgPool(out)

        out = self.conv4(out)

        out += residual
        return out


class BigResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(BigResidualBlockUp, self).__init__()
        self.in_channels = in_channels
        self.temp_channels = max(1, in_channels//4)
        self.out_channels = out_channels
        self.params = in_channels + 3*self.temp_channels
        self.adaDim = spectral_norm(nn.Conv2d(self.in_channels,
                                              self.out_channels,
                                              kernel_size=1, padding=0,
                                              bias=False))

        self.conv1 = spectral_norm(nn.Conv2d(self.in_channels,
                                             self.temp_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.conv2 = spectral_norm(nn.Conv2d(self.temp_channels,
                                             self.temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv3 = spectral_norm(nn.Conv2d(self.temp_channels,
                                             self.temp_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.conv4 = spectral_norm(nn.Conv2d(self.temp_channels,
                                             self.out_channels,
                                             kernel_size=1, padding=0,
                                             bias=False))

        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        # 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'

        self.relu = nn.SELU()

    def forward(self, x, w):
        residual = x
        residual = self.adaDim(self.upsample(residual))

        out = adaIN(x, mean_style=w, std_style=w)
        out = self.relu(x)
        out = self.conv1(out)

        out = adaIN(out, mean_style=w, std_style=w)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv2(out)

        out = adaIN(out, mean_style=w, std_style=w)
        out = self.relu(out)
        out = self.conv3(out)

        out = adaIN(out, mean_style=w, std_style=w)
        out = self.relu(out)
        out = self.conv4(out)

        out += residual
        return out


# ##############
#   Attention  #
# ##############
# class Attention(nn.Module):
#     def __init__(self, in_channels):
#         super(Attention, self).__init__()
#         self.convF = spectral_norm(nn.Conv2d(in_channels, in_channels,
#                                              kernel_size=1, padding=0,
#                                              stride=1,   bias=False))
#         self.convG = spectral_norm(nn.Conv2d(in_channels, in_channels,
#                                              kernel_size=1, padding=0,
#                                              stride=1,   bias=False))
#         self.convH = spectral_norm(nn.Conv2d(in_channels, in_channels,
#                                              kernel_size=1, padding=0,
#                                              stride=1,   bias=False))
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         residual = x
#         f = self.convF(x)
#         g = self.convG(x)
#         h = self.convH(x)
#         attn_map = self.softmax(torch.matmul(f, g))
#         attn = torch.matmul(h, attn_map)
#         return residual + attn


class Attention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(
            nn.Conv2d(in_channel, in_channel//8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(
            nn.Conv2d(in_channel, in_channel//8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(
            nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(
            f_projection.view(B, -1, H*W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H*W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H*W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma*out + x
        return out
