from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch
from settings import LATENT_SIZE, K_SHOT, BATCH_SIZE
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
[36] https://arxiv.org/pdf/1607.08022.pdf
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, learn=True):
        super(ResidualBlock, self).__init__()
        self.adaDimRes = spectral_norm(nn.Conv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=1))
        self.conv1 = spectral_norm(nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False))

        self.conv2 = spectral_norm(nn.Conv2d(out_channels,
                                             out_channels,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False))

        self.relu = nn.ReLU()
        self.in1 = nn.InstanceNorm2d(out_channels, affine=learn)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=learn)

    def forward(self, x):
        residual = x
        residual = self.adaDimRes(residual)
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, learn=True):
        super(ResidualBlockDown, self).__init__()
        self.norm = norm
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))
        self.adaDim = spectral_norm(nn.Conv2d(in_channels,  out_channels,
                                              kernel_size=1, bias=False))

        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool2d(kernel_size=2)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=learn)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=learn)
        F.instance_norm()

    def forward(self, x):
        residual = x
        residual = self.avgPool(self.adaDim(residual))
        out = self.conv1(x)

        if self.norm:
            out = self.in1(out)

        out = self.relu(out)
        out = self.conv2(out)

        if self.norm:
            out = self.in2(out)

        out = self.relu(out)
        out = self.avgPool(out)
        out += residual
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, learn=True):
        super(ResidualBlockUp, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_channels, affine=learn)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=learn)
        self.relu = nn.ReLU()
        self.adaDim = spectral_norm(nn.Conv2d(in_channels,  out_channels,
                                              kernel_size=1,  bias=False))
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels,
                                             kernel_size=3, padding=1,
                                             bias=False))

        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        # 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'

    def forward(self, x):
        residual = x
        residual = self.upsample(self.adaDim(residual))
        out = self.relu(self.in1(x))
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.relu(self.in2(out))
        out = self.conv2(out)
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


# ###############
#    Embedder   #
# ###############
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlockDown(
            K_SHOT*3, 64, norm=False, learn=False)
        self.residual2 = ResidualBlockDown(64, 128, norm=False, learn=False)
        self.residual3 = ResidualBlockDown(128, 256, norm=False, learn=False)
        self.residual4 = ResidualBlockDown(256, 512, norm=False, learn=False)
        self.residual5 = ResidualBlockDown(512, 512, norm=False, learn=False)
        # self.FcWeights = spectral_norm(nn.Linear(512, 1379))
        # self.FcBias = spectral_norm(nn.Linear(512, 1379))
        self.attention = Attention(128)
        self.relu = nn.ReLU()

    def forward(self, x):  # b, 12, 224, 224
        out = self.residual1(x)  # b, 64, 112, 112
        out = self.residual2(out)  # b, 128, 56, 56
        out = self.attention(out)  # b, 128, 56, 56
        out = self.residual3(out)  # b, 256, 28, 28
        out = self.residual4(out)  # b, 512, 14, 14
        out = self.residual5(out)  # b, 512, 7, 7
        out = torch.sum(out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
        out = self.relu(out)
        return out


# ################
#    Generator   #
# ################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Down
        self.ResDown1 = ResidualBlockDown(3, 32, norm=True, learn=False)
        self.ResDown2 = ResidualBlockDown(32, 64, norm=True, learn=False)
        self.ResDown3 = ResidualBlockDown(64, 128, norm=True, learn=False)
        self.attentionDown = Attention(128)
        # Constant
        self.ResBlock_128_1 = ResidualBlock(128, 128, learn=False)
        self.ResBlock_128_2 = ResidualBlock(128, 128, learn=False)
        self.ResBlock_128_3 = ResidualBlock(128, 128, learn=False)
        self.ResBlock_128_4 = ResidualBlock(128, 128, learn=False)
        self.ResBlock_128_5 = ResidualBlock(128, 128, learn=False)
        # Up
        self.ResUp1 = ResidualBlockUp(128, 64, learn=False)
        self.ResUp2 = ResidualBlockUp(64, 32, learn=False)
        self.ResUp3 = ResidualBlockUp(32, 3, learn=False)
        self.attentionUp = Attention(64)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, img):
        # self.ResBlock_128_1.in1.weight = nn.Parameter(
        #     paramNorm["weights"].narrow(1, 0*128, 128))
        # self.ResBlock_128_1.in1.weight = nn.Parameter(
        #     paramNorm["bias"].narrow(1, 0*128, 128))
        # self.ResBlock_128_1.in2.weight = nn.Parameter(
        #     paramNorm["weights"].narrow(1, 1*128, 128))
        # self.ResBlock_128_1.in2.weight = nn.Parameter(
        #     paramNorm["bias"].narrow(1, 1*128, 128))

        x = self.ResDown1(img)
        x = self.ResDown2(x)
        x = self.ResDown3(x)
        x = self.attentionDown(x)

        x = self.ResBlock_128_1(x)  # 2, 128, 55, 55
        x = self.ResBlock_128_2(x)  # 2, 128, 55, 55
        x = self.ResBlock_128_3(x)  # 2, 128, 55, 55
        x = self.ResBlock_128_4(x)  # 2, 128, 55, 55

        x = self.ResBlock_128_5(x)  # 2, 128, 5

        x = self.ResUp1(x)  # 2, 64, 109, 109
        x = self.attentionUp(x)
        x = self.ResUp2(x)
        x = self.ResUp3(x)
        return self.tanh(x)


# ######################
#     Discriminator    #
# ######################
class Discriminator(nn.Module):
    def __init__(self, num_persons, fine_tunning=False):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 64,  norm=True, learn=False)
        self.residual2 = ResidualBlockDown(64, 128, norm=True, learn=False)
        self.residual3 = ResidualBlockDown(128, 256, norm=True, learn=False)
        self.residual4 = ResidualBlockDown(256, 512, norm=True, learn=False)
        self.residual5 = ResidualBlockDown(512, 512, norm=True, learn=False)
        self.attention = Attention(128)
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x, indexes):  # b, 6, 224, 224
        features_maps = []
        out = self.residual1(x)  # b, 64, 112, 112
        features_maps.append(out)
        out = self.residual2(out)  # 2, 128, 56, 56
        features_maps.append(out)
        out = self.attention(out)  # 2, 128, 56, 56
        features_maps.append(out)
        out = self.residual3(out)  # 2, 256, 28, 28
        features_maps.append(out)
        out = self.residual4(out)  # 2, 512, 14, 14
        features_maps.append(out)
        out = self.residual5(out)  # 2, 512, 7,7
        features_maps.append(out)
        out = torch.sum(out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
        out = self.relu(out)
        w0 = self.w0.repeat(BATCH_SIZE).view(BATCH_SIZE, LATENT_SIZE)
        b = self.b.repeat(BATCH_SIZE)
        out = torch.bmm(
            self.embeddings(indexes).view(BATCH_SIZE, 1, LATENT_SIZE),
            (out+w0).view(BATCH_SIZE, LATENT_SIZE, 1)
        )
        out = out.view(BATCH_SIZE)
        out += b
        return out, features_maps


# self.ResBlock_128_2.in1.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 2*128, 128))
# self.ResBlock_128_2.in1.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 2*128, 128))
# self.ResBlock_128_2.in2.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 3*128, 128))
# self.ResBlock_128_2.in2.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 3*128, 128))

# self.ResBlock_128_3.in1.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 4*128, 128))
# self.ResBlock_128_3.in1.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 4*128, 128))
# self.ResBlock_128_3.in2.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 5*128, 128))
# self.ResBlock_128_3.in2.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 5*128, 128))

# self.ResBlock_128_4.in1.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 6*128, 128))
# self.ResBlock_128_4.in1.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 6*128, 128))
# self.ResBlock_128_4.in2.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 7*128, 128))
# self.ResBlock_128_4.in2.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 7*128, 128))

# self.ResBlock_128_5.in1.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 8*128, 128))
# self.ResBlock_128_5.in1.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 8*128, 128))
# self.ResBlock_128_5.in2.weight = nn.Parameter(
#     paramNorm["weights"].narrow(1, 9*128, 128))
# self.ResBlock_128_5.in2.weight = nn.Parameter(
#     paramNorm["bias"].narrow(1, 9*128, 128))
