from torch import nn


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
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.adaDimVonc = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        print("res  ", residual.size())
        residual = self.adaDimVonc(residual)
        print("res  ", residual.size())
        out = self.conv1(x)
        print("conv1  ", out.size())
        out = self.bn1(out)
        print("bn1  ", out.size())
        out = self.relu(out)
        print("relu  ", out.size())
        out = self.conv2(out)
        print("conv2  ", out.size())
        out = self.bn2(out)
        print("bn2  ", out.size())
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlock(48, 64)
        self.residual2 = ResidualBlock(64, 128)
        self.residual3 = ResidualBlock(128, 256)
        self.residual4 = ResidualBlock(256, 512)

    def forward(self, x):
        out = self.residual1(x)
        # out = self.residual2(out)
        # out = self.residual3(out)
        # out = self.residual4(out)
        # out = self.residual5(out)
        print(out.size())
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1_32_9_1 = nn.Conv2d(3, 32, kernel_size=9, stride=1,
                                      padding=1, bias=False)

        self.conv2_64_3_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2,
                                      padding=1, bias=False)

        self.conv3_128_3_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2,
                                       padding=1, bias=False)

        self.ResBlock_128 = ResidualBlock(128, 128)

        self.deconv1_64_3_2 = nn.ConvTranspose2d(128, 64, kernel_size=3,
                                                 stride=2)
        self.deconv2_32_3_2 = nn.ConvTranspose2d(64, 32, kernel_size=3,
                                                 stride=2)
        self.deconv3_3_9_1 = nn.ConvTranspose2d(32, 3, kernel_size=9,
                                                stride=1)

        self.spatial_batchNorm = None
        self.relu = None
        self.tanh = None

    def forward(self, img):
        # TODO
        # TODO
        # TODO CHANGER LES CONV EN RESBLOCK COMME PAPIER LIGNE 51 !
        # TODO
        # TODO

        x = self.relu(self.spatial_batchNorm(self.conv1_32_9_9_1(img)))
        x = self.relu(self.spatial_batchNorm(self.conv1_64_3_3_2(x)))
        x = self.relu(self.spatial_batchNorm(self.conv1_128_3_3_2(x)))
        x = self.ResBlock_128(x)
        x = self.ResBlock_128(x)
        x = self.ResBlock_128(x)
        x = self.ResBlock_128(x)
        x = self.ResBlock_128(x)
        x = self.relu(self.spatial_batchNorm(self.deconv1_64_3_3_2(x)))
        x = self.relu(self.spatial_batchNorm(self.conv1_32_3_3_2(x)))
        x = self.relu(self.spatial_batchNorm(self.conv1_3_9_9_1(x)))
        return None


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()

    def forward(sefl, arg):
        return None


# class Vgg_face(nn.Module):

#     def __init__(self):
#         super(Vgg_face, self).__init__()
#         self.meta = {'mean': [129.186279296875,
#                               104.76238250732422,
#                               93.59396362304688],
#                      'std': [1, 1, 1],
#                      'imageSize': [224, 224, 3]}
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu1_1 = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu1_2 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
#                                   dilation=1, ceil_mode=False)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu2_1 = nn.ReLU(inplace=True)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu2_2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
#                                   dilation=1, ceil_mode=False)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu3_1 = nn.ReLU(inplace=True)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu3_2 = nn.ReLU(inplace=True)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu3_3 = nn.ReLU(inplace=True)
#         self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
#                                   dilation=1, ceil_mode=False)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu4_1 = nn.ReLU(inplace=True)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu4_2 = nn.ReLU(inplace=True)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu4_3 = nn.ReLU(inplace=True)
#         self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
#                                   dilation=1, ceil_mode=False)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu5_1 = nn.ReLU(inplace=True)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu5_2 = nn.ReLU(inplace=True)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1),
#                                  padding=(1, 1))
#         self.relu5_3 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
#                                   dilation=1, ceil_mode=False)
#         self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.dropout6 = nn.Dropout(p=0.5)
#         self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.dropout7 = nn.Dropout(p=0.5)
#         self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

#     def forward(self, x0):
#         x1 = self.conv1_1(x0)
#         x2 = self.relu1_1(x1)
#         x3 = self.conv1_2(x2)
#         x4 = self.relu1_2(x3)
#         x5 = self.pool1(x4)
#         x6 = self.conv2_1(x5)
#         x7 = self.relu2_1(x6)
#         x8 = self.conv2_2(x7)
#         x9 = self.relu2_2(x8)
#         x10 = self.pool2(x9)
#         x11 = self.conv3_1(x10)
#         x12 = self.relu3_1(x11)
#         x13 = self.conv3_2(x12)
#         x14 = self.relu3_2(x13)
#         x15 = self.conv3_3(x14)
#         x16 = self.relu3_3(x15)
#         x17 = self.pool3(x16)
#         x18 = self.conv4_1(x17)
#         x19 = self.relu4_1(x18)
#         x20 = self.conv4_2(x19)
#         x21 = self.relu4_2(x20)
#         x22 = self.conv4_3(x21)
#         x23 = self.relu4_3(x22)
#         x24 = self.pool4(x23)
#         x25 = self.conv5_1(x24)
#         x26 = self.relu5_1(x25)
#         x27 = self.conv5_2(x26)
#         x28 = self.relu5_2(x27)
#         x29 = self.conv5_3(x28)
#         x30 = self.relu5_3(x29)
#         x31_preflatten = self.pool5(x30)
#         x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
#         x32 = self.fc6(x31)
#         x33 = self.relu6(x32)
#         x34 = self.dropout6(x33)
#         x35 = self.fc7(x34)
#         x36 = self.relu7(x35)
#         x37 = self.dropout7(x36)
#         x38 = self.fc8(x37)
#         return x38


# def vgg_face_dag(weights_path=None, **kwargs):
#     """
#     load imported model instance

#     Args:
#         weights_path (str): If set, loads model weights from the given path
#     """
#     model = Vgg_face()
#     if weights_path:
#         state_dict = torch.load(weights_path)
#         model.load_state_dict(state_dict)
#     return model
