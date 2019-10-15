import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

from settings import BATCH_SIZE, LATENT_SIZE, CONCAT, HALF
from utils import load_layers

(ResidualBlock,
 ResidualBlockDown,
 ResidualBlockUp,
 Attention) = load_layers()
# ###############
#    Embedder   #
# ###############


class BigEmbedder(nn.Module):
    """Class for the embedding network

    Arguments:
        None

    Returns:
        Create the model of the network (used then in utils.py -> load_models )
    """

    def __init__(self):
        """
        Initialise the layers
        Layers created for the BIG artchitecture
        Same as model.py but with more layers with wider receptive fields
        All are residuals with spectral norm
        Attention is present on two different size
        fully connected are used to grow the 1*512 to the size of the generator
        """
        super(BigEmbedder, self).__init__()
        self.residual1 = ResidualBlockDown(3, 32)
        self.residual2 = ResidualBlockDown(32, 64)
        self.residual3 = ResidualBlockDown(64, 128)
        self.residual4 = ResidualBlock(128, 128)
        self.attention1 = Attention(128)
        self.residual5 = ResidualBlockDown(128, 256)
        self.residual6 = ResidualBlock(256, 512)
        self.residual7 = ResidualBlockDown(512, LATENT_SIZE)
        self.attention2 = Attention(LATENT_SIZE)
        self.residual8 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        if LATENT_SIZE == 512:
            self.FcWeights = spectral_norm(nn.Linear(LATENT_SIZE, 8014))
            self.FcBias = spectral_norm(nn.Linear(LATENT_SIZE, 8014))
        elif LATENT_SIZE == 1024:
            self.FcWeights = spectral_norm(nn.Linear(LATENT_SIZE, 13390))
            self.FcBias = spectral_norm(nn.Linear(LATENT_SIZE, 13390))
        self.relu = nn.SELU()
        self.avgPool = nn.AvgPool2d(kernel_size=7)

    def forward(self, x):  # b, 12, 224, 224
        """Forward pass :

        The network should take a BATCH picture as input of size (B*3)*W*H
        It takes the pictures ONE by ONE to compute their latent representation
        and then take the mean of all this representation to get the batch one.
        Returns:
            Tensor -- Size 1*512 corresponding to the latent
                                        representation of this BATCH of image
        """
        temp = torch.zeros(LATENT_SIZE,
                           dtype=(torch.half if HALF else torch.float),
                           device="cuda")
        layerUp0 = torch.zeros((BATCH_SIZE, LATENT_SIZE, 7, 7),
                               dtype=(torch.half if HALF else torch.float),
                               device="cuda")
        layerUp1 = torch.zeros((BATCH_SIZE, 512, 14, 14),
                               dtype=(torch.half if HALF else torch.float),
                               device="cuda")
        layerUp2 = torch.zeros((BATCH_SIZE, 256, 14, 14),
                               dtype=(torch.half if HALF else torch.float),
                               device="cuda")
        layerUp3 = torch.zeros((BATCH_SIZE, 128, 28, 28),
                               dtype=(torch.half if HALF else torch.float),
                               device="cuda")

        for i in range(x.size(1)//3):
            out = self.residual1(x.narrow(1, i*3, 3))  # b, 64, 112, 112
            out = self.relu(out)

            out = self.residual2(out)  # b, 128, 56, 56
            out = self.relu(out)

            out = self.residual3(out)  # b, 128, 56, 56
            out = self.relu(out)

            out = self.residual4(out)  # b, 256, 28, 28
            out = self.relu(out)
            # print("L3  ", out.size(), layerUp3.size())
            layerUp3 = torch.add(out, layerUp3)

            out = self.attention1(out)  # b, 128, 56, 56
            out = self.relu(out)

            out = self.residual5(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("L2  ", out.size(), layerUp2.size())
            layerUp2 = torch.add(out, layerUp2)

            out = self.residual6(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("L1  ", out.size(), layerUp1.size())
            layerUp1 = torch.add(out, layerUp1)

            out = self.residual7(out)  # b, 512, 7, 7
            out = self.relu(out)
            # print("L0  ", out.size(), layerUp0.size())
            layerUp0 = torch.add(out, layerUp0)

            out = self.attention2(out)  # b, 512, 14, 14
            out = self.relu(out)
            out = self.residual8(out)
            # print("OUT  ", out.size())

            out = self.avgPool(out).squeeze()
            out = self.relu(out)

            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1)//3))
        layerUp3 = torch.div(layerUp3, (x.size(1)//3))
        layerUp2 = torch.div(layerUp2, (x.size(1)//3))
        layerUp1 = torch.div(layerUp1, (x.size(1)//3))
        layerUp0 = torch.div(layerUp0, (x.size(1)//3))

        paramWeights = self.relu(self.FcWeights(context)).squeeze()
        paramBias = self.relu(self.FcBias(context)).squeeze()

        layersUp = (layerUp0, layerUp1, layerUp2, layerUp3)
        return context, paramWeights, paramBias, layersUp


# ################
#    Generator   #
# ################
class BigGenerator(nn.Module):
    """
    Class for the BigGenerator : It takes ONE landmark image and output a
    synthetic face, helped with layers and coeficient from the embedder.

    Returns:
        Create the model of the network (used then in utils.py -> load_models )
    """

    def __init__(self):
        """
        Layers created for the BIG artchitecture
        Same as model.py but with more layers with wider receptive fields
        All are residuals with spectral norm
        Attention is present on three different size (down constant and up)
        """
        super(BigGenerator, self).__init__()
        # Down
        self.ResDown1 = ResidualBlockDown(3, 32)
        self.ResDown2 = ResidualBlockDown(32, 64)
        self.ResDown3 = ResidualBlockDown(64, 128)
        self.ResDown4 = ResidualBlock(128, 128)
        self.attentionDown1 = Attention(128)
        self.ResDown5 = ResidualBlockDown(128, 256)
        self.ResDown6 = ResidualBlock(256, 512)
        self.ResDown7 = ResidualBlockDown(512, LATENT_SIZE)
        self.attentionDown2 = Attention(LATENT_SIZE)

        # Constant
        self.ResBlock_128_1 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.ResBlock_128_2 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.ResBlock_128_3 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.attention = Attention(LATENT_SIZE)
        self.ResBlock_128_4 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.ResBlock_128_5 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        # Up
        if CONCAT:
            self.Ada0 = spectral_norm(nn.Conv2d(LATENT_SIZE * 2,
                                                LATENT_SIZE,
                                                kernel_size=3,
                                                padding=1, bias=False))
            self.Ada1 = spectral_norm(nn.Conv2d(512 * 2, 512, kernel_size=3,
                                                padding=1, bias=False))
            self.Ada2 = spectral_norm(nn.Conv2d(256*2, 256, kernel_size=3,
                                                padding=1, bias=False))
            self.Ada3 = spectral_norm(nn.Conv2d(128*2, 128, kernel_size=3,
                                                padding=1, bias=False))

        self.Res1 = ResidualBlock(LATENT_SIZE, 512)

        self.ResUp2 = ResidualBlockUp(512, 512)

        self.ResUp3 = ResidualBlock(512, 256)

        self.Res4 = ResidualBlockUp(256, 128)
        self.attentionUp = Attention(128)

        self.ResUp5 = ResidualBlock(128, 64)

        self.ResUp6 = ResidualBlockUp(64, 32)

        self.ResUp7 = ResidualBlockUp(32, 3)
        self.Res8 = ResidualBlockUp(3, 3)

        self.relu = nn.SELU()
        self.tanh = nn.Tanh()

    def forward(self, img, pWeights, pBias, layersUp):
        """
        Res block : in out out out
        Res block up : in out//4 out//4 out//4
        LayersUp are corresponding to the same size layer down of the embedder

        weights and biases are given by the embedder to ponderate the instance
        norm of the constant and upsampling parts.
        It's given in an hard coded bad manner.
        (could be done with loops and be more scalable...
        but I will do it later, it's easier to debug this way)
        """
        layerUp0, layerUp1, layerUp2, layerUp3 = layersUp
        # print("L3 ", layerUp3.size())
        # print("L2 ", layerUp2.size())
        # print("L1 ", layerUp1.size())
        # print("L0 ", layerUp0.size())
        # print("IMG ", img.size())

        # ######
        # DOWN #
        # ######

        x = self.ResDown1(img)
        x = self.relu(x)
        # print("ResDown1  ", x.size())

        x = self.ResDown2(x)
        x = self.relu(x)
        # print("ResDown2  ", x.size())

        x = self.ResDown3(x)
        x = self.relu(x)
        # print("ResDown3  ", x.size())

        x = self.ResDown4(x)
        x = self.relu(x)
        # print("ResDown4  ", x.size())

        if CONCAT == "first":
            x = torch.cat((x, layerUp3), dim=1)
            # print("cat3", x.size())
            x = self.Ada3(x)

        x = self.attentionDown1(x)
        x = self.relu(x)
        # print("ATT1  ", x.size())

        x = self.ResDown5(x)
        x = self.relu(x)
        # print("ResDown5  ", x.size())

        if CONCAT == "first":
            x = torch.cat((x, layerUp2), dim=1)
            # print("cat2", x.size())
            x = self.Ada2(x)

        x = self.ResDown6(x)
        x = self.relu(x)
        # print("ResDown6  ", x.size())

        if CONCAT == "first":
            x = torch.cat((x, layerUp1), dim=1)
            # print("cat1", x.size())
            x = self.Ada1(x)

        x = self.ResDown7(x)
        x = self.relu(x)
        # print("ResDown7  ", x.size())

        if CONCAT == "first":
            x = torch.cat((x, layerUp0), dim=1)
            # print("cat0", x.size())
            x = self.Ada0(x)

        x = self.attentionDown2(x)
        x = self.relu(x)
        # print("ATT2  ", x.size())

        # ##########
        # CONSTANT #
        # ##########

        i = 0

        nb_params = self.ResBlock_128_1.params
        x = self.ResBlock_128_1(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_1  ", x.size())
        i += nb_params

        nb_params = self.ResBlock_128_2.params
        x = self.ResBlock_128_2(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_2  ", x.size())
        i += nb_params

        nb_params = self.ResBlock_128_3.params
        x = self.ResBlock_128_3(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_3  ", x.size())
        i += nb_params

        x = self.attention(x)
        x = self.relu(x)

        nb_params = self.ResBlock_128_4.params
        x = self.ResBlock_128_4(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_4  ", x.size())
        i += nb_params

        nb_params = self.ResBlock_128_5.params
        x = self.ResBlock_128_5(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_5  ", x.size())
        i += nb_params

        # ####
        # UP #
        # ####

        nb_params = self.Res1.params
        x = self.Res1(x, w=pWeights.narrow(-1, i, nb_params),
                      b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("Res1  ", x.size())
        i += nb_params

        if CONCAT == "last":
            x = torch.cat((x, layerUp0), dim=1)
            x = self.Ada0(x)
            x = self.relu(x)

        nb_params = self.ResUp2.params
        x = self.ResUp2(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp2  ", x.size())
        i += nb_params

        if CONCAT == "last":
            x = torch.cat((x, layerUp1), dim=1)
            x = self.Ada1(x)
            x = self.relu(x)

        nb_params = self.ResUp3.params
        x = self.ResUp3(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp3  ", x.size())
        i += nb_params

        if CONCAT == "last":
            x = torch.cat((x, layerUp2), dim=1)
            x = self.Ada2(x)
            x = self.relu(x)

        nb_params = self.Res4.params
        x = self.Res4(x, w=pWeights.narrow(-1, i, nb_params),
                      b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("Res4  ", x.size())
        i += nb_params

        if CONCAT == "last":
            x = torch.cat((x, layerUp3), dim=1)
            x = self.Ada3(x)
            x = self.relu(x)

        x = self.attentionUp(x)
        x = self.relu(x)

        nb_params = self.ResUp5.params
        x = self.ResUp5(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp5  ", x.size())
        i += nb_params

        nb_params = self.ResUp6.params
        x = self.ResUp6(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp6  ", x.size())
        i += nb_params

        nb_params = self.ResUp7.params
        x = self.ResUp7(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp7  ", x.size())
        i += nb_params

        nb_params = self.Res8.params
        x = self.Res8(x, w=pWeights.narrow(-1, i, nb_params),
                      b=pBias.narrow(-1, i, nb_params))
        x = self.tanh(x)
        # print("Res8  ", x.size())
        i += nb_params
        # print("Nb_param   ", i)
        return x


# ######################
#     Discriminator    #
# ######################
class BigDiscriminator(nn.Module):
    """
    Class for the BigDiscriminator
    Architecture is almost the same as the embedder.

    Arguments:
        num_persons {int} -- The number of persons in the dataset. It's used to
        create the embeddings for each persons.
    Returns:
        Create the model of the network (used then in utils.py -> load_models )
    """

    def __init__(self, num_persons, fine_tunning=False):
        """[summary]

        Arguments:
        num_persons {int} -- The number of persons in the dataset. It's used to
        Create the embeddings for each persons.

        Keyword Arguments:
            fine_tunning {bool} -- will be used after... still not implemented
            (default: {False})
            Will be used to prevent the loading of embeddings to fintune only
            on one unknown person (variables are differents).
        """
        super(BigDiscriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 32)
        self.residual2 = ResidualBlockDown(32, 64)
        self.residual3 = ResidualBlockDown(64, 128)
        self.residual4 = ResidualBlock(128, 128)
        self.attention1 = Attention(128)
        self.residual5 = ResidualBlockDown(128, 256)
        self.residual6 = ResidualBlock(256, 256)
        self.residual7 = ResidualBlockDown(256, 512)
        self.attention2 = Attention(512)
        self.residual8 = ResidualBlock(512, LATENT_SIZE)
        self.residual9 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.SELU()
        self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))
        self.sig = nn.Sigmoid()
        self.avgPool = nn.AvgPool2d(kernel_size=7)

    def forward(self, x, indexes):
        features_maps = []
        out = self.residual1(x)
        out = self.relu(out)
        # print("Out 1 ", out.size())
        features_maps.append(out)

        out = self.residual2(out)
        out = self.relu(out)
        # print("Out 2 ", out.size())
        features_maps.append(out)

        out = self.residual3(out)
        out = self.relu(out)
        # print("Out 3 ", out.size())
        features_maps.append(out)

        out = self.residual4(out)
        out = self.relu(out)
        # print("Out 4 ", out.size())
        features_maps.append(out)

        out = self.attention1(out)
        out = self.relu(out)
        # print("Out 11 ", out.size())
        features_maps.append(out)

        out = self.residual5(out)
        out = self.relu(out)
        # print("Out 5 ", out.size())
        features_maps.append(out)

        out = self.residual6(out)
        out = self.relu(out)
        # print("Out 6 ", out.size())
        features_maps.append(out)

        out = self.residual7(out)
        out = self.relu(out)
        # print("Out 7 ", out.size())
        features_maps.append(out)

        out = self.attention2(out)
        out = self.relu(out)
        # print("Out 22 ", out.size())
        features_maps.append(out)

        out = self.residual8(out)
        out = self.relu(out)
        # print("Out 8 ", out.size())
        features_maps.append(out)

        out = self.residual9(out)
        out = self.relu(out)
        # print("Out 9 ", out.size())
        features_maps.append(out)

        out = self.avgPool(out).squeeze()
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
        final_out = self.sig(final_out)
        return final_out, features_maps
