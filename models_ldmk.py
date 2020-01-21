import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from settings import ATTENTION, BATCH_SIZE, CONCAT, DEVICE, HALF, LATENT_SIZE
from utils import load_layers

# (ResidualBlock, ResidualBlockDown, ResidualBlockUp, Attention) = load_layers()

# # ################
# #    Generator   #
# # ################

# Size of z latent vector (i.e. size of generator input)
nz = 10

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(10, 32, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(32),
            nn.SELU(True),
            nn.ConvTranspose2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.SELU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.SELU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(64),
            nn.SELU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(32),
            nn.SELU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(32, 3, 4, 2, 3, bias=False),
            nn.InstanceNorm2d(3),
            nn.SELU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.SELU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.SELU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.SELU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.SELU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
        )
        self.fc = nn.Linear(11*11, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        out = self.conv(input)
        out = self.fc(out.flatten(1))
        out = self.sig(out)
        return out


# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
#         self.conv4 = nn.Conv2d(32, 3, 3, padding=1)
#         self.in1 = nn.InstanceNorm2d(32, affine=True)
#         self.in2 = nn.InstanceNorm2d(64, affine=True)
#         self.in3 = nn.InstanceNorm2d(32, affine=True)
#         self.tanh = nn.Tanh()
#         self.relu = nn.SELU()

#     def forward(self, x):
#         out = self.in1(self.relu(self.conv1(x)))
#         out = self.in2(self.relu(self.conv2(out)))
#         out = self.in3(self.relu(self.conv3(out)))
#         out = self.conv4(out)
#         out = self.tanh(out)
#         return out


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, stride=(2, 2))
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=(2, 2))
#         self.conv3 = nn.Conv2d(64, 32, 3, stride=(2, 2))
#         self.conv4 = nn.Conv2d(32, 3, 3, stride=(2, 2))
#         self.fc = nn.Linear(3*13*13, 1)
#         self.sig = nn.Sigmoid()
#         self.relu = nn.SELU()

#     def forward(self, x):
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.relu(self.conv3(out))
#         out = self.relu(self.conv4(out))
#         # print(out.size())
#         out = self.fc(out.flatten(1))
#         # print(out.size())
#         out = self.sig(out)
#         return out


# class Generator(nn.Module):
#     """
#     Class for the BigGenerator : It takes ONE landmark image and output a
#     synthetic face, helped with layers and coeficient from the embedder.

#     Returns:
#         Create the model of the network (used then in utils.py -> load_models )
#     """

#     def __init__(self):
#         """
#         Layers created for the BIG artchitecture
#         Same as model.py but with more layers with wider receptive fields
#         All are residuals with spectral norm
#         Attention is present on three different size (down constant and up)
#         """
#         super().__init__()
#         # Down
#         self.ResDown1 = ResidualBlockDown(1, 64)  # 64, 112
#         self.ResDown2 = ResidualBlockDown(64, 128)  # 128, 64
#         if ATTENTION:
#             self.attentionDown1 = Attention(128)
#         self.ResDown3 = ResidualBlockDown(128, 256)  # 256, 32
#         self.ResDown4 = ResidualBlockDown(256, LATENT_SIZE)  # 512, 16
#         if ATTENTION:
#             self.attentionDown2 = Attention(LATENT_SIZE)
#         # Constant
#         self.ResBlock_1 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16
#         self.ResBlock_2 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16
#         if ATTENTION:
#             self.attention1 = Attention(LATENT_SIZE)
#         # Up
#         self.ResUp1 = ResidualBlockUp(LATENT_SIZE, 256)   # 256, 32
#         self.ResUp2 = ResidualBlockUp(256, 128)   # 128, 64
#         if ATTENTION:
#             self.attentionUp1 = Attention(128)
#         self.ResUp3 = ResidualBlockUp(128, 64)  # 64, 112, 112
#         self.ResUp4 = ResidualBlockUp(64, 3)  # 3, 224, 224
#         if ATTENTION:
#             self.attentionUp2 = Attention(3)
#         self.conv1 = spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
#         self.conv2 = spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
#         self.relu = nn.SELU()
#         self.tanh = nn.Tanh()

#     def forward(self, img):
#         """
#         Res block : in out out out
#         Res block up : in out//4 out//4 out//4
#         LayersUp are corresponding to the same size layer down of the embedder

#         weights and biases are given by the embedder to ponderate the instance
#         norm of the constant and upsampling parts.
#         It's given in an hard coded bad manner.
#         (could be done with loops and be more scalable...
#         but I will do it later, it's easier to debug this way)
#         """

#         # ######
#         # DOWN #
#         # ######

#         x = self.ResDown1(img)
#         x = self.relu(x)
#         # print("ResDown1  ", x.size())

#         x = self.ResDown2(x)
#         x = self.relu(x)
#         # print("ResDown2  ", x.size())

#         if ATTENTION:
#             x = self.attentionDown1(x)
#             x = self.relu(x)

#         x = self.ResDown3(x)
#         x = self.relu(x)
#         # print("ResDown3  ", x.size())

#         x = self.ResDown4(x)
#         x = self.relu(x)
#         if ATTENTION:
#             x = self.attentionDown2(x)
#             x = self.relu(x)
#         # print("ResDown4  ", x.size())

#         # ##########
#         # CONSTANT #
#         # ##########
#         x = self.ResBlock_1(x)
#         x = self.relu(x)

#         x = self.ResBlock_2(x)
#         x = self.relu(x)

#         if ATTENTION:
#             x = self.attention1(x)
#             x = self.relu(x)

#         # ####
#         # UP #
#         # ####

#         x = self.ResUp1(x)
#         x = self.relu(x)
#         # print("Res1  ", x.size())
#         x = self.ResUp2(x)
#         x = self.relu(x)
#         # print("ResUp2  ", x.size())

#         if ATTENTION:
#             x = self.attentionUp1(x)
#             x = self.relu(x)

#         x = self.ResUp3(x)
#         x = self.relu(x)
#         # print("ResUp3  ", x.size())

#         x = self.ResUp4(x)
#         x = self.relu(x)
#         # print("Res4  ", x.size())

#         if ATTENTION:
#             x = self.attentionUp2(x)
#             x = self.relu(x)

#         x = F.instance_norm(x)
#         x = self.relu(x)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.tanh(x)
#         return x


# # ######################
# #     Discriminator    #
# # ######################
# class Discriminator(nn.Module):
#     """
#     Class for the BigDiscriminator
#     Architecture is almost the same as the embedder.

#     Arguments:
#         num_persons {int} -- The number of persons in the dataset. It's used to
#         create the embeddings for each persons.
#     Returns:
#         Create the model of the network (used then in utils.py -> load_models )
#     """

#     def __init__(self, num_persons, fine_tunning=False):
#         """[summary]

#         Arguments:
#         num_persons {int} -- The number of persons in the dataset. It's used to
#         Create the embeddings for each persons.

#         Keyword Arguments:
#             fine_tunning {bool} -- will be used after... still not implemented
#             (default: {False})
#             Will be used to prevent the loading of embeddings to fintune only
#             on one unknown person (variables are differents).
#         """
#         super().__init__()
#         self.residual1 = ResidualBlock(3, 64)  # 224
#         self.residual2 = ResidualBlockDown(64, 128)  # 224
#         self.residual3 = ResidualBlockDown(128, 256)  # 112
#         self.attention1 = Attention(256)
#         self.residual4 = ResidualBlockDown(256, LATENT_SIZE)  # 66
#         self.residual5 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 33
#         self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 16
#         self.attention2 = Attention(LATENT_SIZE)
#         self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
#         self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
#         self.b = nn.Parameter(torch.rand(1), requires_grad=True)
#         self.relu = nn.SELU()
#         self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))
#         self.tanh = nn.Tanh()
#         self.sig = nn.Sigmoid()
#         self.avgPool = nn.AvgPool2d(kernel_size=7)

#     def forward(self, x):
#         features_maps = []
#         out = self.residual1(x)
#         # print("Out 1 ", out.size())
#         features_maps.append(out)

#         out = self.residual2(out)
#         out = self.relu(out)
#         # print("Out 2 ", out.size())
#         features_maps.append(out)

#         out = self.residual3(out)
#         out = self.relu(out)
#         # print("Out 3 ", out.size())
#         features_maps.append(out)

#         out = self.attention1(out)
#         out = self.relu(out)
#         features_maps.append(out)

#         out = self.residual4(out)
#         out = self.relu(out)
#         # print("Out 4 ", out.size())
#         features_maps.append(out)

#         out = self.residual5(out)
#         out = self.relu(out)
#         # print("Out 5 ", out.size())
#         features_maps.append(out)

#         out = self.residual6(out)
#         out = self.relu(out)
#         # print("Out 6 ", out.size())
#         features_maps.append(out)

#         out = self.attention2(out)
#         out = self.relu(out)
#         # print("Out 22 ", out.size())
#         features_maps.append(out)

#         out = self.avgPool(out).squeeze()
#         out = self.relu(out)
#         final_out = self.fc(out)
#         features_maps.append(out)

#         # final_out = self.sig(final_out)
#         return final_out, features_maps
