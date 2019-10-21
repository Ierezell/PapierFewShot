from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch

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

    def forward(self, x, w=None, b=None):
        if w is not None and b is not None:
            w1 = w.narrow(-1, 0, self.in_channels)
            b1 = b.narrow(-1, 0, self.in_channels)
            w2 = w.narrow(-1, self.in_channels, self.temp_channels)
            b2 = b.narrow(-1, self.in_channels, self.temp_channels)
            w3 = w.narrow(-1, self.in_channels +
                          self.temp_channels, self.temp_channels)
            b3 = b.narrow(-1, self.in_channels +
                          self.temp_channels, self.temp_channels)
            w4 = w.narrow(-1, self.in_channels+2 *
                          self.temp_channels, self.temp_channels)
            b4 = b.narrow(-1, self.in_channels+2 *
                          self.temp_channels, self.temp_channels)

        residual = self.relu(self.adaDim(x))

        x = F.instance_norm(x)
        if w is not None and b is not None:
            out = w1.unsqueeze(-1).unsqueeze(-1).expand_as(x) * x
            out = out + b1.unsqueeze(-1).unsqueeze(-1).expand_as(out)
        else:
            out = x

        out = self.relu(out)
        out = self.conv1(x)

        out = F.instance_norm(out)
        if w is not None and b is not None:
            out = w2.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
            out = out + b2.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv2(out)

        out = F.instance_norm(out)
        if w is not None and b is not None:
            out = w3.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
            out = out + b3.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv3(out)

        out = F.instance_norm(out)
        if w is not None and b is not None:
            out = w4.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
            out = out + b4.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv4(out)

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
        if hasattr(self, "adaDim"):
            # fill_to_out_channels = self.adaDim(residual)
            # residual = torch.cat((residual, fill_to_out_channels), dim=1)
            residual = self.adaDim(residual)

        out = F.instance_norm(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.instance_norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = F.instance_norm(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = F.instance_norm(out)
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

    def forward(self, x, w, b):
        w1 = w.narrow(-1, 0, self.in_channels)
        b1 = b.narrow(-1, 0, self.in_channels)
        w2 = w.narrow(-1, self.in_channels, self.temp_channels)
        b2 = b.narrow(-1, self.in_channels, self.temp_channels)
        w3 = w.narrow(-1, self.in_channels +
                      self.temp_channels, self.temp_channels)
        b3 = b.narrow(-1, self.in_channels +
                      self.temp_channels, self.temp_channels)
        w4 = w.narrow(-1, self.in_channels+2 *
                      self.temp_channels, self.temp_channels)
        b4 = b.narrow(-1, self.in_channels+2 *
                      self.temp_channels, self.temp_channels)
        x = F.instance_norm(x)
        residual = x
        residual = self.adaDim(self.upsample(residual))

        out = w1.unsqueeze(-1).unsqueeze(-1).expand_as(x) * x
        out = out + b1.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv1(out)

        out = F.instance_norm(out)
        out = w2.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
        out = out + b2.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv2(out)

        out = F.instance_norm(out)
        out = w3.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
        out = out + b3.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv3(out)

        out = F.instance_norm(out)
        out = w4.unsqueeze(-1).unsqueeze(-1).expand_as(out) * out
        out = out + b4.unsqueeze(-1).unsqueeze(-1).expand_as(out)

        out = self.relu(out)
        out = self.conv4(out)
        out += residual
        return out


# ##############
#   Attention  #
# ##############
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.convF = spectral_norm(nn.Conv2d(in_channels, in_channels,
                                             kernel_size=1, padding=0,
                                             stride=1,   bias=False))
        self.convG = spectral_norm(nn.Conv2d(in_channels, in_channels,
                                             kernel_size=1, padding=0,
                                             stride=1,   bias=False))
        self.convH = spectral_norm(nn.Conv2d(in_channels, in_channels,
                                             kernel_size=1, padding=0,
                                             stride=1,   bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = x
        f = self.convF(x)
        g = self.convG(x)
        h = self.convH(x)
        attn_map = self.softmax(torch.matmul(f, g))
        attn = torch.matmul(h, attn_map)
        return residual + attn
