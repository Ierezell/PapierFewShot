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
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlockDown(128, 256)
        self.residual4 = ResidualBlockDown(256, 512)
        self.residual5 = ResidualBlockDown(512, 512)
        self.FcWeights = spectral_norm(nn.Linear(512, 2952))
        self.FcBias = spectral_norm(nn.Linear(512, 2952))
        self.attention = Attention(128)
        self.relu = nn.ReLU()

    def forward(self, x):  # b, 12, 224, 224
        temp = torch.tensor(np.zeros(LATENT_SIZE, dtype=np.float),
                            dtype=torch.float, device="cuda")
        for i in range(x.size(1)//3):
            out = self.residual1(x.narrow(1, i*3, 3))  # b, 64, 112, 112
            out = self.relu(out)
            out = self.residual2(out)  # b, 128, 56, 56
            out = self.relu(out)
            out = self.attention(out)  # b, 128, 56, 56
            out = self.relu(out)
            out = self.residual3(out)  # b, 256, 28, 28
            out = self.relu(out)
            out = self.residual4(out)  # b, 512, 14, 14
            out = self.relu(out)
            out = self.residual5(out)  # b, 512, 7, 7
            out = self.relu(out)
            out = torch.sum(
                out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
            out = self.relu(out)
            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1)//3))
        paramWeights = self.FcWeights(out).squeeze()
        paramBias = self.FcBias(out).squeeze()
        return context, paramWeights, paramBias


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
        self.attentionDown = Attention(128)
        # Constant
        self.ResBlock_128_1 = ResidualBlock(128, 128)
        self.ResBlock_128_2 = ResidualBlock(128, 128)
        self.ResBlock_128_3 = ResidualBlock(128, 128)
        self.ResBlock_128_4 = ResidualBlock(128, 128)
        self.ResBlock_128_5 = ResidualBlock(128, 128)
        # Up
        self.ResUp1 = ResidualBlockUp(128, 64)
        self.ResUp2 = ResidualBlockUp(64, 32)
        self.ResUp3 = ResidualBlockUp(32, 3)
        self.attentionUp = Attention(64)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img,  paramWeights, paramBias):
        x = self.ResDown1(img)
        x = self.relu(x)
        x = self.ResDown2(x)
        x = self.relu(x)
        x = self.ResDown3(x)
        x = self.relu(x)
        x = self.attentionDown(x)
        x = self.relu(x)

        x = self.ResBlock_128_1(x,
                                w1=paramWeights.narrow(-1, 0*128, 128),
                                b1=paramBias.narrow(-1, 0 * 128, 128),
                                w2=paramWeights.narrow(-1, 1*128, 128),
                                b2=paramBias.narrow(-1, 1 * 128, 128),
                                w3=paramWeights.narrow(-1, 2*128, 128),
                                b3=paramBias.narrow(-1, 2*128, 128),
                                w4=paramWeights.narrow(-1, 3*128, 128),
                                b4=paramBias.narrow(-1, 3*128, 128)
                                )
        # TODO Register backward hook

        x = self.relu(x)
        # b, 128, 55, 55

        x = self.ResBlock_128_2(x,
                                w1=paramWeights.narrow(-1, 4*128, 128),
                                b1=paramBias.narrow(-1, 4*128, 128),
                                w2=paramWeights.narrow(-1, 5*128, 128),
                                b2=paramBias.narrow(-1, 5*128, 128),
                                w3=paramWeights.narrow(-1, 6*128, 128),
                                b3=paramBias.narrow(-1, 6*128, 128),
                                w4=paramWeights.narrow(-1, 7*128, 128),
                                b4=paramBias.narrow(-1, 7*128, 128),
                                )
        x = self.relu(x)
        # b, 128, 55, 55

        x = self.ResBlock_128_3(x,
                                w1=paramWeights.narrow(-1, 8*128, 128),
                                b1=paramBias.narrow(-1, 8*128, 128),
                                w2=paramWeights.narrow(-1, 9*128, 128),
                                b2=paramBias.narrow(-1, 9*128, 128),
                                w3=paramWeights.narrow(-1, 10*128, 128),
                                b3=paramBias.narrow(-1, 10*128, 128),
                                w4=paramWeights.narrow(-1, 11*128, 128),
                                b4=paramBias.narrow(-1, 11*128, 128),
                                )
        x = self.relu(x)
        # b, 128, 55, 55

        x = self.ResBlock_128_4(x,
                                w1=paramWeights.narrow(-1, 12*128, 128),
                                b1=paramBias.narrow(-1, 12*128, 128),
                                w2=paramWeights.narrow(-1, 13*128, 128),
                                b2=paramBias.narrow(-1, 13*128, 128),
                                w3=paramWeights.narrow(-1, 14*128, 128),
                                b3=paramBias.narrow(-1, 14*128, 128),
                                w4=paramWeights.narrow(-1, 15*128, 128),
                                b4=paramBias.narrow(-1, 15*128, 128),
                                )
        x = self.relu(x)
        # b, 128, 55, 55

        x = self.ResBlock_128_5(x,
                                w1=paramWeights.narrow(-1, 16*128, 128),
                                b1=paramBias.narrow(-1, 16*128, 128),
                                w2=paramWeights.narrow(-1, 17*128, 128),
                                b2=paramBias.narrow(-1, 17*128, 128),
                                w3=paramWeights.narrow(-1, 18*128, 128),
                                b3=paramBias.narrow(-1, 18*128, 128),
                                w4=paramWeights.narrow(-1, 19*128, 128),
                                b4=paramBias.narrow(-1, 19*128, 128),
                                )
        x = self.relu(x)
        # b, 128, 5

        x = self.ResUp1(x,
                        w1=paramWeights.narrow(-1, 20*128, 128),
                        b1=paramBias.narrow(-1, 20*128, 128),
                        w2=paramWeights.narrow(-1, 21*128, 128//4),
                        b2=paramBias.narrow(-1, 21*128, 128//4),
                        w3=paramWeights.narrow(-1, 21*128+128//4, 128//4),
                        b3=paramBias.narrow(-1, 21*128+128//4, 128//4),
                        w4=paramWeights.narrow(-1, 21*128+128//2, 128//4),
                        b4=paramBias.narrow(-1, 21*128+128//2, 128//4),
                        )
        x = self.relu(x)
        # b, 64, 109, 109

        x = self.attentionUp(x)
        x = self.relu(x)

        o_r1 = 21*128+128//2+128//4
        x = self.ResUp2(x,
                        w1=paramWeights.narrow(-1, o_r1, 64),
                        b1=paramBias.narrow(-1, o_r1, 64),
                        w2=paramWeights.narrow(-1, o_r1+64, 64//4),
                        b2=paramBias.narrow(-1, o_r1+64, 64//4),
                        w3=paramWeights.narrow(-1, o_r1+64+64//4, 64//4),
                        b3=paramBias.narrow(-1, o_r1+64+64//4, 64//4),
                        w4=paramWeights.narrow(-1, o_r1+64+64//2, 64//4),
                        b4=paramBias.narrow(-1, o_r1+64+64//2, 64//4),
                        )
        x = self.relu(x)
        o_r2 = o_r1+64+64//2+64//4
        x = self.ResUp3(x,
                        w1=paramWeights.narrow(-1, o_r2, 32),
                        b1=paramBias.narrow(-1, o_r2, 32),
                        w2=paramWeights.narrow(-1, o_r2+32, 32//4),
                        b2=paramBias.narrow(-1, o_r2+32, 32//4),
                        w3=paramWeights.narrow(-1, o_r2+32+32//4, 32//4),
                        b3=paramBias.narrow(-1, o_r2+32+32//4, 32//4),
                        w4=paramWeights.narrow(-1, o_r2+32+32//2, 32//4),
                        b4=paramBias.narrow(-1, o_r2+32+32//2, 32//4),
                        )
        # print("Taille poids et bias ", o_r2+32+32//2+32//4)
        x = self.sigmoid(x)
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
        self.residual4 = ResidualBlockDown(256, 512)
        self.residual5 = ResidualBlockDown(512, 512)
        self.residual6 = ResidualBlockDown(512, 512)
        self.attention1 = Attention(128)
        self.attention2 = Attention(512)
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.ReLU()
        self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))

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
        return final_out, features_maps
