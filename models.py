from settings import LATENT_SIZE, BATCH_SIZE, CONCAT, DEVICE
from torch.nn.utils import spectral_norm
from torch import nn
import torch
import numpy as np
from utils import load_layers

(ResidualBlock,
 ResidualBlockDown,
 ResidualBlockUp,
 Attention) = load_layers()
# ###############
#    Embedder   #
# ###############


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlockDown(3, 32)
        self.residual2 = ResidualBlockDown(32, 64)
        self.residual3 = ResidualBlockDown(64, 128)
        self.residual4 = ResidualBlockDown(128, 256)
        self.residual5 = ResidualBlockDown(256, LATENT_SIZE)
        # self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)
        self.FcWeights = spectral_norm(nn.Linear(LATENT_SIZE, 1960))
        self.FcBias = spectral_norm(nn.Linear(LATENT_SIZE, 1960))
        self.attention = Attention(64)
        self.avgPool = torch.nn.AvgPool2d(7)
        self.relu = nn.SELU()

    def forward(self, x):  # b, 12, 224, 224
        temp = torch.zeros(LATENT_SIZE, dtype=torch.float, device=DEVICE)

        layerUp0 = torch.zeros((BATCH_SIZE, LATENT_SIZE, 7, 7),
                               dtype=torch.float, device=DEVICE)
        layerUp1 = torch.zeros((BATCH_SIZE, 256, 14, 14),
                               dtype=torch.float, device=DEVICE)
        layerUp2 = torch.zeros((BATCH_SIZE, 128, 28, 28),
                               dtype=torch.float, device=DEVICE)
        layerUp3 = torch.zeros((BATCH_SIZE, 64, 56, 56),
                               dtype=torch.float, device=DEVICE)

        for i in range(x.size(1) // 3):
            # print("x  ", x.size())
            out = self.residual1(x.narrow(1, i*3, 3))  # b, 64, 112, 112
            out = self.relu(out)
            # print("out1  ", out.size())

            out = self.residual2(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("out2  ", out.size())

            out = self.attention(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("out4  ", out.size())
            layerUp3 = torch.add(out, layerUp3)

            out = self.residual3(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("out3  ", out.size())

            layerUp2 = torch.add(out, layerUp2)

            out = self.residual4(out)  # b, 256, 28, 28
            out = self.relu(out)
            # print("out5 ", out.size())
            layerUp1 = torch.add(out, layerUp1)

            out = self.residual5(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("out6  ", out.size())
            layerUp0 = torch.add(out, layerUp0)
            out = self.avgPool(out).squeeze()  # b,512
            # print("out  ", out.size())
            out = self.relu(out)
            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1) // 3))

        layerUp0 = torch.div(layerUp0, (x.size(1) // 3))
        layerUp1 = torch.div(layerUp1, (x.size(1) // 3))
        layerUp2 = torch.div(layerUp2, (x.size(1) // 3))
        layerUp3 = torch.div(layerUp3, (x.size(1) // 3))

        paramWeights = self.FcWeights(context)
        paramBias = self.FcBias(context)
        layersUp = (layerUp0, layerUp1, layerUp2, layerUp3)

        return context, paramWeights, paramBias, layersUp


# ################
#    Generator   #
# ################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Down
        self.ResDown1 = ResidualBlockDown(3, 32)
        self.ResDown2 = ResidualBlockDown(32, 64)
        self.attentionDown = Attention(64)
        self.ResDown3 = ResidualBlockDown(64, 128)
        self.ResDown4 = ResidualBlockDown(128, 256)
        self.ResDown5 = ResidualBlockDown(256, LATENT_SIZE)
        # Constant
        # self.ResBlock_128_1 = ResidualBlock(128, 128)
        # self.ResBlock_128_2 = ResidualBlock(128, 256)
        self.ResBlock_128_3 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        # Up
        self.ResUp1 = ResidualBlockUp(LATENT_SIZE, 256)
        self.ResUp2 = ResidualBlockUp(256, 128)
        self.ResUp3 = ResidualBlockUp(128, 64)
        self.attentionUp = Attention(64)
        self.ResUp4 = ResidualBlockUp(64, 32)
        self.ResUp5 = ResidualBlockUp(32, 3)

        if CONCAT:
            self.Ada0 = spectral_norm(nn.Conv2d(LATENT_SIZE * 2, LATENT_SIZE,
                                                kernel_size=3,
                                                padding=1, bias=False))
            self.Ada1 = spectral_norm(nn.Conv2d(256 * 2, 256,
                                                kernel_size=3,
                                                padding=1, bias=False))
            self.Ada2 = spectral_norm(nn.Conv2d(128 * 2, 128, kernel_size=3,
                                                padding=1, bias=False))
            self.Ada3 = spectral_norm(nn.Conv2d(64*2, 64, kernel_size=3,
                                                padding=1, bias=False))
        self.relu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, pWeights, pBias, layersUp):
        (layerUp0, layerUp1, layerUp2, layerUp3) = layersUp
        # print(layerUp1.size())
        # print(layerUp2.size())
        # print(layerUp3.size())
        # print(img.size())

        # ######
        # DOWN #
        # ######
        x = self.ResDown1(img)
        x = self.relu(x)
        # print("resdown1", x.size())

        x = self.ResDown2(x)
        x = self.relu(x)
        # print("resdown2", x.size())

        x = self.attentionDown(x)
        x = self.relu(x)

        if "first" in CONCAT:
            x = torch.cat((x, layerUp3), dim=1)
            # print("cat3", x.size())
            x = self.Ada3(x)

        x = self.ResDown3(x)
        x = self.relu(x)
        # print("resdown3", x.size())

        if "first" in CONCAT:
            x = torch.cat((x, layerUp2), dim=1)
            # print("cat2", x.size())
            x = self.Ada2(x)

        x = self.ResDown4(x)
        x = self.relu(x)
        # print("resdown3", x.size())

        if "first" in CONCAT:
            x = torch.cat((x, layerUp1), dim=1)
            # print("cat1", x.size())
            x = self.Ada1(x)

        x = self.ResDown5(x)
        x = self.relu(x)

        # ##########
        # CONSTANT #
        # ##########

        i = 0
        nb_params = self.ResBlock_128_3.params
        x = self.ResBlock_128_3(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        i += nb_params
        # print("ResBlock3", x.size())    # b, 128, 55, 55

        if "middle" in CONCAT:
            x = torch.cat((x, layerUp0), dim=1)
            x = self.Ada0(x)
            x = self.relu(x)

        # ####
        # Up #
        # ####

        nb_params = self.ResUp1.params
        x = self.ResUp1(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        i += nb_params
        # print("ResUp1", x.size())
        if "last" in CONCAT:
            x = torch.cat((x, layerUp1), dim=1)
            x = self.Ada1(x)
            x = self.relu(x)

        nb_params = self.ResUp2.params
        x = self.ResUp2(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        i += nb_params
        # print("ResUp2", x.size())

        if "last" in CONCAT:
            x = torch.cat((x, layerUp2), dim=1)
            x = self.Ada2(x)
            x = self.relu(x)

        nb_params = self.ResUp3.params
        x = self.ResUp3(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        i += nb_params
        # print("ResUp3", x.size())

        if "last" in CONCAT:
            x = torch.cat((x, layerUp3), dim=1)
            x = self.Ada3(x)
            x = self.relu(x)

        x = self.attentionUp(x)
        x = self.relu(x)
        # print("AttUp", x.size())

        nb_params = self.ResUp4.params
        x = self.ResUp4(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        i += nb_params
        # print("ResUp4", x.size())

        nb_params = self.ResUp5.params
        x = self.ResUp5(x, pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.tanh(x)
        i += nb_params
        # print("ResUp5", x.size())

        # print("Nb_param   ", i)
        return x


# ######################
#     Discriminator    #
# ######################
class Discriminator(nn.Module):
    def __init__(self, num_persons, fine_tunning=False):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 32)
        self.residual2 = ResidualBlockDown(32, 64)
        self.residual3 = ResidualBlockDown(64, 128)
        self.attention1 = Attention(128)
        self.residual4 = ResidualBlockDown(128, 256)
        self.attention2 = Attention(256)
        self.residual5 = ResidualBlockDown(256, LATENT_SIZE)
        self.avgPool = torch.nn.AvgPool2d(7)
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.SELU()
        self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, indexes):  # b, 6, 224, 224
        features_maps = []
        out = self.residual1(x)  # b, 64, 112, 112
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual2(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual3(out)  # 2, 256, 28, 28
        out = self.relu(out)
        features_maps.append(out)

        out = self.attention1(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual4(out)  # 2, 512, 14, 14
        out = self.relu(out)
        features_maps.append(out)

        out = self.attention2(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual5(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = self.avgPool(out).squeeze()  # b,512
        out = self.relu(out)
        final_out = self.fc(out)

        w0 = self.w0.repeat(BATCH_SIZE).view(BATCH_SIZE, LATENT_SIZE)
        b = self.b.repeat(BATCH_SIZE)

        condition = torch.bmm(
            self.embeddings(indexes).view(-1, 1, LATENT_SIZE),
            (out+w0).view(BATCH_SIZE, LATENT_SIZE, 1)
        )
        final_out += condition.view(final_out.size())
        final_out = final_out.view(b.size())
        final_out += b
        final_out = self.sig(final_out)
        return final_out, features_maps
