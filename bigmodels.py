import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

from settings import BATCH_SIZE, LATENT_SIZE, CONCAT, HALF, DEVICE
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
        self.residual1 = ResidualBlock(6, 64)
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlockDown(128, 256)
        self.attention1 = Attention(256)
        self.residual4 = ResidualBlockDown(256, LATENT_SIZE)
        self.residual5 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)
        self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)
        self.attention2 = Attention(LATENT_SIZE)

        self.FcWeights = spectral_norm(nn.Linear(LATENT_SIZE, 8017))
        self.FcBias = spectral_norm(nn.Linear(LATENT_SIZE, 8017))
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
                           device=DEVICE)
        # layerUp0 = torch.zeros((BATCH_SIZE, LATENT_SIZE, 7, 7),
        #                        dtype=(torch.half if HALF else torch.float),
        #                        device=DEVICE)
        # layerUp1 = torch.zeros((BATCH_SIZE, 512, 14, 14),
        #                        dtype=(torch.half if HALF else torch.float),
        #                        device=DEVICE)
        # layerUp2 = torch.zeros((BATCH_SIZE, 256, 14, 14),
        #                        dtype=(torch.half if HALF else torch.float),
        #                        device=DEVICE)
        # layerUp3 = torch.zeros((BATCH_SIZE, 128, 28, 28),
        #                        dtype=(torch.half if HALF else torch.float),
        #                        device=DEVICE)

        for i in range(x.size(1)//6):
            # print(x.narrow(1, i*6, 6).size())
            out = self.residual1(x.narrow(1, i*6, 6))  # b, 64, 112, 112
            # print("1 ", out.size())
            out = self.residual2(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("2 ", out.size())

            out = self.residual3(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("3 ", out.size())

            out = self.attention1(out)  # b, 128, 56, 56
            out = self.relu(out)
            # print("4 ", out.size())

            out = self.residual4(out)  # b, 256, 28, 28
            out = self.relu(out)
            # print("5 ", out.size())
            # print("L3  ", out.size(), layerUp3.size())
            # layerUp3 = torch.add(out, layerUp3)

            out = self.residual5(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("6 ", out.size())
            # print("1", "L2  ", out.size(), layerUp2.size())
            # layerUp2 = torch.add(out, layerUp2)

            out = self.residual6(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("7 ", out.size())
            # print("L1  ", out.size(), layerUp1.size())
            # layerUp1 = torch.add(out, layerUp1)

            out = self.attention2(out)  # b, 512, 14, 14
            out = self.relu(out)
            # print("OUT  ", out.size())

            out = self.avgPool(out).squeeze()
            out = self.relu(out)

            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1)//6))
        # layerUp3 = torch.div(layerUp3, (x.size(1)//3))
        # layerUp2 = torch.div(layerUp2, (x.size(1)//3))
        # layerUp1 = torch.div(layerUp1, (x.size(1)//3))
        # layerUp0 = torch.div(layerUp0, (x.size(1)//3))

        paramWeights = self.relu(self.FcWeights(context)).squeeze()
        paramBias = self.relu(self.FcBias(context)).squeeze()

        # layersUp = (layerUp0, layerUp1, layerUp2, layerUp3)
        layersUp = (None, None, None, None)
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
        self.ResDown1 = ResidualBlockDown(3, 64)
        self.ResDown2 = ResidualBlockDown(64, 128)
        self.ResDown3 = ResidualBlockDown(128, 256)
        self.attentionDown = Attention(256)
        self.ResDown4 = ResidualBlockDown(256, LATENT_SIZE)
        # Constant
        self.ResBlock_128_1 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.ResBlock_128_2 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.attention1 = Attention(LATENT_SIZE)
        self.ResBlock_128_3 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.ResBlock_128_4 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)
        self.attention2 = Attention(LATENT_SIZE)
        # Up
        self.ResUp1 = ResidualBlockUp(LATENT_SIZE, 256)
        self.ResUp2 = ResidualBlockUp(256, 128)
        self.attentionUp1 = Attention(128)
        self.ResUp3 = ResidualBlockUp(128, 64)
        self.ResUp4 = ResidualBlockUp(64, 3)
        self.attentionUp2 = Attention(3)
        self.conv = spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
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

        x = self.attentionDown(x)
        x = self.relu(x)

        x = self.ResDown4(x)
        x = self.relu(x)
        # print("ResDown4  ", x.size())

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

        x = self.attention1(x)
        x = self.relu(x)

        nb_params = self.ResBlock_128_3.params
        x = self.ResBlock_128_3(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_3  ", x.size())
        i += nb_params

        nb_params = self.ResBlock_128_4.params
        x = self.ResBlock_128_4(x, w=pWeights.narrow(-1, i, nb_params),
                                b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResBlock_128_4  ", x.size())
        i += nb_params

        x = self.attention2(x)
        x = self.relu(x)

        # ####
        # UP #
        # ####

        nb_params = self.ResUp1.params
        x = self.ResUp1(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("Res1  ", x.size())
        i += nb_params

        nb_params = self.ResUp2.params
        x = self.ResUp2(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp2  ", x.size())
        i += nb_params

        x = self.attentionUp1(x)
        x = self.relu(x)

        nb_params = self.ResUp3.params
        x = self.ResUp3(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("ResUp3  ", x.size())
        i += nb_params

        nb_params = self.ResUp4.params
        x = self.ResUp4(x, w=pWeights.narrow(-1, i, nb_params),
                        b=pBias.narrow(-1, i, nb_params))
        x = self.relu(x)
        # print("Res4  ", x.size())
        i += nb_params

        x = self.attentionUp2(x)
        x = self.relu(x)
        x = self.instance_norm(x)
        w = pWeights.narrow(-1, 0, 3)
        b = pBias.narrow(-1, 0, 3)

        x = w.unsqueeze(-1).unsqueeze(-1).expand_as(x) * x
        x = x + b.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
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
        self.residual1 = ResidualBlock(6, 64)  # 224
        self.residual2 = ResidualBlockDown(64, 128)  # 224
        self.residual3 = ResidualBlockDown(128, 256)  # 112
        self.attention1 = Attention(256)
        self.residual4 = ResidualBlockDown(256, LATENT_SIZE)  # 66
        self.residual5 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 33
        self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 16
        self.attention2 = Attention(LATENT_SIZE)
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

        out = self.attention1(out)
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual4(out)
        out = self.relu(out)
        # print("Out 4 ", out.size())
        features_maps.append(out)

        out = self.residual5(out)
        out = self.relu(out)
        # print("Out 5 ", out.size())
        features_maps.append(out)

        out = self.residual6(out)
        out = self.relu(out)
        # print("Out 6 ", out.size())
        features_maps.append(out)

        out = self.attention2(out)
        out = self.relu(out)
        # print("Out 22 ", out.size())
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
