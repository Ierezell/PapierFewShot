from settings import LATENT_SIZE, BATCH_SIZE
from layers import ResidualBlock, ResidualBlockDown, ResidualBlockUp, Attention
from torch.nn.utils import spectral_norm
from torch import nn
import torch
import numpy as np


# ###############
#    Embedder   #
# ###############


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlockDown(3, 64)
        # self.residual2 = ResidualBlock(32, 64)
        self.residual3 = ResidualBlockDown(64, 128)
        self.residual4 = ResidualBlockDown(128, 256)
        self.residual5 = ResidualBlockDown(256, 512)
        self.residual6 = ResidualBlockDown(512, 512)
        self.FcWeights = spectral_norm(nn.Linear(512, 3385))
        self.FcBias = spectral_norm(nn.Linear(512, 3385))
        self.attention = Attention(128)
        self.relu = nn.SELU()

    def forward(self, x):  # b, 12, 224, 224
        temp = torch.tensor(np.zeros(LATENT_SIZE, dtype=np.float),
                            dtype=torch.float, device="cuda")

        layerUp1 = torch.tensor(np.zeros((512, 14, 14), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp2 = torch.tensor(np.zeros((256, 28, 28), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp3 = torch.tensor(np.zeros((128, 56, 56), dtype=np.float),
                                dtype=torch.float, device="cuda")

        for i in range(x.size(1) // 3):
            out = self.residual1(x.narrow(1, i*3, 3))  # b, 64, 112, 112
            out = self.relu(out)

            # out = self.residual2(out)  # b, 128, 56, 56
            # out = self.relu(out)

            out = self.residual3(out)  # b, 128, 56, 56
            out = self.relu(out)

            out = self.attention(out)  # b, 128, 56, 56
            out = self.relu(out)
            layerUp3 = torch.add(out, layerUp3)

            out = self.residual4(out)  # b, 256, 28, 28
            out = self.relu(out)
            layerUp2 = torch.add(out, layerUp2)

            out = self.residual5(out)  # b, 512, 14, 14
            out = self.relu(out)
            layerUp1 = torch.add(out, layerUp1)

            out = self.residual6(out)  # b, 512, 7, 7
            out = self.relu(out)
            out = torch.sum(
                out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
            out = self.relu(out)
            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1)//3))
        paramWeights = self.FcWeights(out).squeeze()
        paramBias = self.FcBias(out).squeeze()
        layersUp = (layerUp1, layerUp2, layerUp3)

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
        self.ResDown3 = ResidualBlockDown(64, 128)
        self.ResDown4 = ResidualBlockDown(128, 128)
        # self.attentionDown = Attention(128)
        # Constant
        # self.ResBlock_128_1 = ResidualBlock(128, 128)
        # self.ResBlock_128_2 = ResidualBlock(128, 256)
        self.ResBlock_128_3 = ResidualBlock(128, 512)
        # Up
        self.ResUp1 = ResidualBlockUp(512, 256)
        self.ResUp2 = ResidualBlockUp(256, 128)
        self.ResUp3 = ResidualBlockUp(128, 64)
        self.ResUp4 = ResidualBlockUp(64, 32)
        self.Res5 = ResidualBlock(32, 3)
        # self.attentionUp = Attention(64)

        # self.Ada1 = spectral_norm(nn.Conv2d(512 * 2, 512, kernel_size=3,
        #                                     padding=1, bias=False))
        # self.Ada2 = spectral_norm(nn.Conv2d(256 * 2, 256, kernel_size=3,
        #                                     padding=1, bias=False))
        # self.Ada3 = spectral_norm(nn.Conv2d(128 * 2, 128, kernel_size=3,
        #                                     padding=1, bias=False))
        self.relu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, pWeights, pBias, layersUp):
        (layerUp1, layerUp2, layerUp3) = layersUp

        # print(img.size())
        x = self.ResDown1(img)
        x = self.relu(x)
        # print("resdown1", x.size())
        x = self.ResDown2(x)
        x = self.relu(x)
        # print("resdown2", x.size())
        x = self.ResDown3(x)
        x = self.relu(x)
        # print("resdown2", x.size())
        # x = self.attentionDown(x)
        # x = self.relu(x)
        # print("attDown", x.size())
        x = self.ResDown4(x)
        x = self.relu(x)
        # print("resdown3", x.size())
        # TODO Register backward hook
        i = 0
        # i_c = self.ResBlock_128_1.in_channels
        # o_c = self.ResBlock_128_1.out_channels
        # x = self.ResBlock_128_1(x,
        #                         w1=pWeights.narrow(-1, i, i_c),
        #                         b1=pBias.narrow(-1, i, i_c),
        #                         w2=pWeights.narrow(-1, i+i_c, o_c),
        #                         b2=pBias.narrow(-1, i+i_c, o_c),
        #                         w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
        #                         b3=pBias.narrow(-1, i+i_c+o_c, o_c),
        #                         w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
        #                         b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
        #                         )
        # x = self.relu(x)
        # i = i + i_c + (3*o_c)
        # print("ResBlock1", x.size())

        # i_c = self.ResBlock_128_2.in_channels
        # o_c = self.ResBlock_128_2.out_channels
        # x = self.ResBlock_128_2(x,
        #                         w1=pWeights.narrow(-1, i, i_c),
        #                         b1=pBias.narrow(-1, i, i_c),
        #                         w2=pWeights.narrow(-1, i+i_c, o_c),
        #                         b2=pBias.narrow(-1, i+i_c, o_c),
        #                         w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
        #                         b3=pBias.narrow(-1, i+i_c+o_c, o_c),
        #                         w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
        #                         b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
        #                         )
        # x = self.relu(x)
        # i = i + i_c + (3*o_c)        # b, 128, 55, 55
        # print("ResBlock2", x.size())

        i_c = self.ResBlock_128_3.in_channels
        o_c = self.ResBlock_128_3.out_channels
        x = self.ResBlock_128_3(x,
                                w1=pWeights.narrow(-1, i, i_c),
                                b1=pBias.narrow(-1, i, i_c),
                                w2=pWeights.narrow(-1, i+i_c, o_c),
                                b2=pBias.narrow(-1, i+i_c, o_c),
                                w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                                b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                                w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                                b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                                )
        x = self.relu(x)
        i = i + i_c + (3*o_c)
        # print("ResBlock3", x.size())    # b, 128, 55, 55

        # x = torch.cat((x, layerUp1), dim=1)
        # x = self.Ada1(x)
        # x = self.relu(x)

        i_c = self.ResUp1.in_channels
        o_c = self.ResUp1.temp_channels
        x = self.ResUp1(x,
                        w1=pWeights.narrow(-1, i, i_c),
                        b1=pBias.narrow(-1, i, i_c),
                        w2=pWeights.narrow(-1, i+i_c, o_c),
                        b2=pBias.narrow(-1, i+i_c, o_c),
                        w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                        b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                        w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                        b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                        )
        x = self.relu(x)
        i = i + i_c + (3*o_c)
        # print("ResUp1", x.size())

        # x = torch.cat((x, layerUp2), dim=1)
        # x = self.Ada2(x)
        # x = self.relu(x)

        # b, 64, 109, 109

        i_c = self.ResUp2.in_channels
        o_c = self.ResUp2.temp_channels
        x = self.ResUp2(x,
                        w1=pWeights.narrow(-1, i, i_c),
                        b1=pBias.narrow(-1, i, i_c),
                        w2=pWeights.narrow(-1, i+i_c, o_c),
                        b2=pBias.narrow(-1, i+i_c, o_c),
                        w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                        b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                        w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                        b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                        )
        x = self.relu(x)
        i = i + i_c + (3*o_c)
        # print("ResUp2", x.size())
        # print("Layer3 ", layerUp3.size())

        # x = torch.cat((x, layerUp3), dim=1)
        # x = self.Ada3(x)
        # x = self.relu(x)

        i_c = self.ResUp3.in_channels
        o_c = self.ResUp3.temp_channels
        x = self.ResUp3(x,
                        w1=pWeights.narrow(-1, i, i_c),
                        b1=pBias.narrow(-1, i, i_c),
                        w2=pWeights.narrow(-1, i+i_c, o_c),
                        b2=pBias.narrow(-1, i+i_c, o_c),
                        w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                        b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                        w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                        b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                        )
        x = self.relu(x)
        i = i + i_c + (3*o_c)
        # print("ResUp3", x.size())

        # x = self.attentionUp(x)
        # x = self.relu(x)
        # print("AttUp", x.size())

        i_c = self.ResUp4.in_channels
        o_c = self.ResUp4.temp_channels
        x = self.ResUp4(x,
                        w1=pWeights.narrow(-1, i, i_c),
                        b1=pBias.narrow(-1, i, i_c),
                        w2=pWeights.narrow(-1, i+i_c, o_c),
                        b2=pBias.narrow(-1, i+i_c, o_c),
                        w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                        b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                        w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                        b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                        )
        x = self.relu(x)
        i = i + i_c + (3 * o_c)
        # print("ResUp4", x.size())

        i_c = self.Res5.in_channels
        o_c = self.Res5.out_channels
        x = self.Res5(x,
                      w1=pWeights.narrow(-1, i, i_c),
                      b1=pBias.narrow(-1, i, i_c),
                      w2=pWeights.narrow(-1, i+i_c, o_c),
                      b2=pBias.narrow(-1, i+i_c, o_c),
                      w3=pWeights.narrow(-1, i+i_c+o_c, o_c),
                      b3=pBias.narrow(-1, i+i_c+o_c, o_c),
                      w4=pWeights.narrow(-1, i+i_c+(2*o_c), o_c),
                      b4=pBias.narrow(-1, i+i_c+(2*o_c), o_c)
                      )
        x = self.tanh(x)
        i = i + i_c + (3*o_c)
        # print("ResUp5", x.size())

        # print("Nb_param   ", i)
        return x


# ######################
#     Discriminator    #
# ######################
class Discriminator(nn.Module):
    def __init__(self, num_persons, fine_tunning=False):
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 64,)
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlockDown(128, 256)
        self.attention1 = Attention(128)
        self.residual4 = ResidualBlockDown(256, 512)
        self.attention2 = Attention(512)
        self.residual5 = ResidualBlockDown(512, 512)
        self.residual6 = ResidualBlockDown(512, 512)
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.SELU()
        self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))
        self.tanh = nn.Tanh()

    def forward(self, x, indexes):  # b, 6, 224, 224
        features_maps = []
        out = self.residual1(x)  # b, 64, 112, 112
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual2(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.attention1(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual3(out)  # 2, 256, 28, 28
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

        out = self.residual6(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = torch.sum(out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
        out = self.relu(out)
        final_out = self.fc(out)
        features_maps.append(out)

        w0 = self.w0.repeat(BATCH_SIZE).view(BATCH_SIZE, LATENT_SIZE)
        b = self.b.repeat(BATCH_SIZE)

        condition = torch.bmm(
            self.embeddings(indexes).view(-1, 1, LATENT_SIZE),
            (out+w0).view(BATCH_SIZE, LATENT_SIZE, 1)
        )
        final_out += condition.view(final_out.size())
        final_out = final_out.view(b.size())
        final_out += b
        # final_out = self.tanh(final_out)
        final_out /= 10
        return final_out, features_maps
