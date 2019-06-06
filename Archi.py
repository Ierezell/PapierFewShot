from torch import nn
import torch
from settings import LATENT_SIZE
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


TODO W est un embedding, le disciminateur prédit un vecteur que l'on fait
ensuite produit scalaire avec l'embedding de la video i.
TODO mettre l'attenttion
TODO mettre les resblock au lieu des convs

"""


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.adaDimVonc = nn.utils.spectral_norm(nn.Conv2d(in_channels,
                                                           out_channels,
                                                           kernel_size=1)
                                                 )
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels,
                                                      out_channels,
                                                      kernel_size=3, padding=1,
                                                      bias=False)
                                            )
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels,
                                                      out_channels,
                                                      kernel_size=3, padding=1,
                                                      bias=False)
                                            )
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        residual = x
        residual = self.adaDimVonc(residual)
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(ResidualBlockDown, self).__init__()
        self.norm = norm
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels,
                                                      out_channels,
                                                      kernel_size=3, padding=1,
                                                      bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels,
                                                      out_channels,
                                                      kernel_size=3, padding=1,
                                                      bias=False))
        self.adaDim = nn.utils.spectral_norm(nn.Conv2d(in_channels,
                                                       out_channels,
                                                       kernel_size=1,
                                                       bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.avgPool = nn.AvgPool2d(kernel_size=3)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)

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
        # out = self.relu(out)
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockUp, self).__init__()
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels*2,
                                                      out_channels,
                                                      kernel_size=3,
                                                      bias=False))
        self.adaDim = nn.utils.spectral_norm(nn.Conv2d(in_channels*2,
                                                       out_channels,
                                                       kernel_size=1,
                                                       bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels,
                                                      out_channels,
                                                      kernel_size=3,
                                                      bias=False))
        self.upsampleConvX = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, padding=0, stride=2))

        self.upsampleConvRes = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, padding=0, stride=2))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'

    def forward(self, x):
        residual = x
        residual = self.upsampleConvX(self.adaDim(residual))
        out = self.relu(self.in1(x))
        out = self.upsampleConvRes(out)
        out = self.conv1(out)
        out = self.relu(self.in2(x))
        out = self.conv2(out)
        out += residual
        # out = self.relu(out)
        return out

# ##############
#   Attention  #
# ##############


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.convF = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels,
                                                      kernel_size=1,
                                                      padding=0, stride=1,
                                                      bias=False)
                                            )
        self.convG = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels,
                                                      kernel_size=1,
                                                      padding=0, stride=1,
                                                      bias=False)
                                            )
        self.convH = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels,
                                                      kernel_size=1,
                                                      padding=0, stride=1,
                                                      bias=False)
                                            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.convF(x)
        g = self.convG(x)
        h = self.convH(x)

        attn_map = self.softmax(torch.bmm(f, g))
        attn = torch.bmm(h, attn_map)
        return x + attn


# ###############
#    Embedder   #
# ###############


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlockDown(48, 64)
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlockDown(128, 256)
        self.residual4 = ResidualBlockDown(256, 512)
        self.residual5 = ResidualBlockDown(512, 512)

    def forward(self, x):
        # 2, 48, 224, 224
        out = self.residual1(x)
        # 2, 64, 74, 74
        out = self.residual2(out)
        # 2, 128, 24, 24
        out = self.residual3(out)
        # 2, 256, 8, 8
        out = self.residual4(out)
        # 2, 512, 2, 2
        out = self.residual5(out)
        # 2, 512, 1, 1
        out = out.squeeze()
        return out

# ################
#    Generator   #
# ################


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Down
        self.conv1_32_9_1 = nn.utils.spectral_norm(nn.Conv2d(3, 32,
                                                             kernel_size=9,
                                                             stride=1,
                                                             padding=1,
                                                             bias=False)
                                                   )
        self.Norm1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2_64_3_2 = nn.utils.spectral_norm(nn.Conv2d(32, 64,
                                                             kernel_size=3,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False)
                                                   )
        self.Norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3_128_3_2 = nn.utils.spectral_norm(nn.Conv2d(64, 128,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=1,
                                                              bias=False)
                                                    )
        self.Norm3 = nn.InstanceNorm2d(128, affine=True)
        # Constant
        self.ResBlock_128 = ResidualBlock(128, 128)
        self.NormRes = nn.InstanceNorm2d(128, affine=True)
        # Up
        self.deconv1_64_3_2 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(128, 64,
                               kernel_size=3, stride=2, padding=1)
        )
        self.Norm4 = nn.InstanceNorm2d(64, affine=True)

        self.deconv2_32_3_2 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(64, 32,
                               kernel_size=3, stride=2, padding=1)
        )
        self.Norm5 = nn.InstanceNorm2d(32, affine=True)

        self.deconv3_3_9_1 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=0))
        self.Norm6 = nn.InstanceNorm2d(3, affine=True)
        # affine dit sic'est entrainable ou pas

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, img):
        # TODO CHANGER LES CONV EN RESBLOCK COMME PAPIER LIGNE 51 !
        # 2, 3, 224, 224
        x = self.relu(self.Norm1(self.conv1_32_9_1(img)))
        # 2, 32, 218, 218
        x = self.relu(self.Norm2(self.conv2_64_3_2(x)))
        # 2, 64, 109, 109
        x = self.relu(self.Norm3(self.conv3_128_3_2(x)))
        # 2, 128, 55, 55
        x = self.ResBlock_128(x)
        # 2, 128, 55, 55
        x = self.ResBlock_128(x)
        # 2, 128, 55, 55
        x = self.ResBlock_128(x)
        # 2, 128, 55, 55
        x = self.ResBlock_128(x)
        # 2, 128, 55, 55
        x = self.ResBlock_128(x)
        # 2, 128, 55, 55
        # self.Norm4.weight = norm_weights[0:128]
        # self.Norm4.bias = norm_weights[128:256]
        x = self.relu(self.Norm4(self.deconv1_64_3_2(x)))
        # 2, 64, 109, 109
        # self.Norm5.weight = norm_weights[0:32]
        # self.Norm5.weight = norm_weights[0:32]
        x = self.relu(self.Norm5(self.deconv2_32_3_2(x)))
        # 2, 32, 217, 217
        # self.Norm6.weight = norm_weights[0:32]
        # self.Norm6.weight = norm_weights[0:32]
        x = self.relu(self.Norm6(self.deconv3_3_9_1(x)))
        # 2, 3, 225, 225
        return x

# ######################
#     Discriminator    #
# ######################


class Discriminator(nn.Module):
    def __init__(self, num_persons, fine_tunning=False):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 64, norm=False)
        self.residual2 = ResidualBlockDown(64, 128, norm=False)
        self.residual3 = ResidualBlockDown(128, 256, norm=False)
        self.residual4 = ResidualBlockDown(256, 512,  norm=False)
        self.residual5 = ResidualBlockDown(512, 512,  norm=False)
        self.embeddings = nn.Embedding(num_persons, 512)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x, indexes):
        features_maps = []
        2, 6, 224, 224
        out = self.residual1(x)
        features_maps.append(out)
        2, 64, 74, 74
        out = self.residual2(out)
        features_maps.append(out)
        2, 64, 74, 74
        out = self.residual3(out)
        features_maps.append(out)
        2, 128, 24, 24
        out = self.residual4(out)
        features_maps.append(out)
        2, 256, 8, 8
        out = self.residual5(out).squeeze()
        features_maps.append(out)
        2, 512, 2, 2
        w0 = self.w0.repeat(x.size(0)).view(x.size(0), LATENT_SIZE)
        print("W0 : ", w0.size())
        print("out : ", out.size())
        print("out+w0", (out+w0).size())
        print("embeddinghzeu : ", self.embeddings.size())
        out = torch.bmm(out+w0, self.embeddings(indexes))
        print("WAZAAAAAAAAAA : ",out.size())
        out += self.b
        print(out.size())
        return out, features_maps
