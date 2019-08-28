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
        super(Embedder, self).__init__()
        self.residual1 = ResidualBlockDown(3, 64)
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlock(128, 128)
        self.residual4 = ResidualBlockDown(128, 256)
        self.residual5 = ResidualBlockDown(256, 512)
        self.residual6 = ResidualBlock(512, 512)
        self.residual7 = ResidualBlockDown(512, 512)
        self.FcWeights = spectral_norm(nn.Linear(512, 11024))
        self.FcBias = spectral_norm(nn.Linear(512, 11024))
        self.attention1 = Attention(128)
        self.attention2 = Attention(512)
        self.relu = nn.ReLU()

    def forward(self, x):  # b, 12, 224, 224
        """Forward pass :

        The network should take a BATCH picture as input of size (B*3)*W*H
        It takes the pictures ONE by ONE to compute their latent representation
        and then take the mean of all this representation to get the batch one.
        Returns:
            Tensor -- Size 1*512 corresponding to the latent
                                        representation of this BATCH of image
        """
        temp = torch.tensor(np.zeros(LATENT_SIZE, dtype=np.float),
                            dtype=torch.float, device="cuda")
        layerUp1 = torch.tensor(np.zeros((64, 112, 112), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp2 = torch.tensor(np.zeros((128, 56, 56), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp3 = torch.tensor(np.zeros((128, 56, 56), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp4 = torch.tensor(np.zeros((256, 28, 28), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp5 = torch.tensor(np.zeros((512, 14, 14), dtype=np.float),
                                dtype=torch.float, device="cuda")
        layerUp6 = torch.tensor(np.zeros((512, 14, 14), dtype=np.float),
                                dtype=torch.float, device="cuda")

        for i in range(x.size(1)//3):
            out = self.residual1(x.narrow(1, i*3, 3))  # b, 64, 112, 112
            out = self.relu(out)
            layerUp1 = torch.add(out, layerUp1)

            out = self.residual2(out)  # b, 128, 56, 56
            out = self.relu(out)
            layerUp2 = torch.add(out, layerUp2)

            out = self.attention1(out)  # b, 128, 56, 56
            out = self.relu(out)
            out = self.residual3(out)  # b, 128, 56, 56
            out = self.relu(out)
            layerUp3 = torch.add(out, layerUp3)

            out = self.residual4(out)  # b, 256, 28, 28
            out = self.relu(out)
            layerUp4 = torch.add(out, layerUp4)

            out = self.residual5(out)  # b, 512, 14, 14
            out = self.relu(out)
            layerUp5 = torch.add(out, layerUp5)

            out = self.residual6(out)  # b, 512, 14, 14
            out = self.relu(out)
            layerUp6 = torch.add(out, layerUp6)

            out = self.attention2(out)  # b, 512, 14, 14
            out = self.relu(out)
            out = self.residual7(out)  # b, 512, 7, 7
            out = self.relu(out)
            out = torch.sum(
                out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
            out = self.relu(out)
            temp = torch.add(out, temp)

        context = torch.div(temp, (x.size(1)//3))
        layerUp1 = torch.div(layerUp1, (x.size(1)//3))
        layerUp2 = torch.div(layerUp2, (x.size(1)//3))
        layerUp3 = torch.div(layerUp3, (x.size(1)//3))
        layerUp4 = torch.div(layerUp4, (x.size(1)//3))
        layerUp5 = torch.div(layerUp5, (x.size(1)//3))
        layerUp6 = torch.div(layerUp6, (x.size(1)//3))

        paramWeights = self.relu(self.FcWeights(out)).squeeze()
        paramBias = self.relu(self.FcBias(out)).squeeze()
        layersUp = (layerUp1, layerUp2, layerUp3, layerUp4, layerUp5, layerUp6)
        return context, paramWeights, paramBias, layersUp


# ################
#    Generator   #
# ################
class Generator(nn.Module):
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
        super(Generator, self).__init__()
        # Down
        self.ResDown1 = ResidualBlockDown(3, 32)
        self.ResDown2 = ResidualBlockDown(32, 64)
        self.ResDown3 = ResidualBlockDown(64, 128)
        self.ResDown4 = ResidualBlock(128, 128)
        self.attentionDown = Attention(128)
        # Constant
        self.ResBlock_128_1 = ResidualBlock(128, 256)
        self.ResBlock_128_2 = ResidualBlock(256, 512)
        self.ResBlock_128_3 = ResidualBlock(512, 512)
        self.attention = Attention(512)
        self.ResBlock_128_4 = ResidualBlock(512, 512)
        self.ResBlock_128_5 = ResidualBlockDown(512, 512)
        # Up
        self.ResAda1 = ResidualBlock(1024, 512)
        self.ResUp1 = ResidualBlock(512, 512)

        self.ResAda2 = ResidualBlock(1024, 512)
        self.Res2 = ResidualBlockUp(512, 256)

        self.ResAda3 = ResidualBlock(512, 256)
        self.ResUp3 = ResidualBlockUp(256, 128)

        self.ResAda4 = ResidualBlock(256, 128)
        self.ResUp4 = ResidualBlock(128, 128)

        self.ResAda5 = ResidualBlock(256, 128)
        self.ResUp5 = ResidualBlockUp(128, 64)

        self.ResAda6 = ResidualBlock(128, 64)
        self.Res6 = ResidualBlockUp(64, 32)

        self.Res7 = ResidualBlock(32, 3)
        self.attentionUp = Attention(64)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, paramWeights, paramBias, layersUp):
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
        layerUp1, layerUp2, layerUp3, layerUp4, layerUp5, layerUp6 = layersUp
        x = self.ResDown1(img)
        x = self.relu(x)
        x = self.ResDown2(x)
        x = self.relu(x)
        x = self.ResDown3(x)
        x = self.relu(x)
        x = self.ResDown4(x)
        x = self.relu(x)
        x = self.attentionDown(x)
        x = self.relu(x)

        x = self.ResBlock_128_1(x,
                                w1=paramWeights.narrow(-1, 0, 128),
                                b1=paramBias.narrow(-1, 0, 128),
                                w2=paramWeights.narrow(-1, 128, 256),
                                b2=paramBias.narrow(-1, 128, 256),
                                w3=paramWeights.narrow(-1, 128+256, 256),
                                b3=paramBias.narrow(-1, 128+256, 256),
                                w4=paramWeights.narrow(-1, 128+256+256, 256),
                                b4=paramBias.narrow(-1, 128+256+256, 256)
                                )
        x = self.relu(x)
        outR1 = 128 + 256 + 256 + 256

        # TODO Register backward hook

        x = self.ResBlock_128_2(
            x,
            w1=paramWeights.narrow(-1, outR1, 256),
            b1=paramBias.narrow(-1, outR1, 256),
            w2=paramWeights.narrow(-1, outR1+256, 512),
            b2=paramBias.narrow(-1, outR1+256, 512),
            w3=paramWeights.narrow(-1, outR1+256+512, 512),
            b3=paramBias.narrow(-1, outR1+256+512, 512),
            w4=paramWeights.narrow(-1, outR1+256+512+512, 512),
            b4=paramBias.narrow(-1, outR1+256+512+512, 512)
        )
        x = self.relu(x)
        outR2 = outR1 + 256 + 512 + 512 + 512

        x = self.ResBlock_128_3(
            x,
            w1=paramWeights.narrow(-1, outR2, 512),
            b1=paramBias.narrow(-1, outR2, 512),
            w2=paramWeights.narrow(-1, outR2+512, 512),
            b2=paramBias.narrow(-1, outR2+512, 512),
            w3=paramWeights.narrow(-1, outR2+512+512, 512),
            b3=paramBias.narrow(-1, outR2+512+512, 512),
            w4=paramWeights.narrow(-1, outR2+512+512+512, 512),
            b4=paramBias.narrow(-1, outR2+512+512+512, 512))
        x = self.relu(x)
        outR3 = outR2 + 512 + 512 + 512 + 512

        x = self.attention(x)
        x = self.relu(x)

        x = self.ResBlock_128_4(
            x,
            w1=paramWeights.narrow(-1, outR3, 512),
            b1=paramBias.narrow(-1, outR3, 512),
            w2=paramWeights.narrow(-1, outR3+512, 512),
            b2=paramBias.narrow(-1, outR3+512, 512),
            w3=paramWeights.narrow(-1, outR3+512+512, 512),
            b3=paramBias.narrow(-1, outR3+512+512, 512),
            w4=paramWeights.narrow(-1,  outR3+512+512+512, 512),
            b4=paramBias.narrow(-1, outR3 + 512 + 512 + 512, 512))
        x = self.relu(x)
        outR4 = outR3 + 512 + 512 + 512 + 512
        x = self.ResBlock_128_5(x)
        x = self.relu(x)
        outR5 = outR4

        x = torch.cat((x, layerUp6), dim=1)
        x = self.ResAda1(x)
        x = self.relu(x)
        x = self.ResUp1(x,
                        w1=paramWeights.narrow(-1, outR5, 512),
                        b1=paramBias.narrow(-1, outR5, 512),
                        w2=paramWeights.narrow(-1, outR5+512, 512),
                        b2=paramBias.narrow(-1, outR5+512, 512),
                        w3=paramWeights.narrow(-1, outR5+512+512, 512),
                        b3=paramBias.narrow(-1, outR5+512+512, 512),
                        w4=paramWeights.narrow(-1, outR5+512+512+512, 512),
                        b4=paramBias.narrow(-1, outR5 + 512 + 512 + 512, 512))
        x = self.relu(x)
        outRu1 = outR5 + 512 + 512 + 512 + 512

        x = torch.cat((x, layerUp5), dim=1)
        x = self.ResAda2(x)
        x = self.relu(x)
        x = self.Res2(
            x,
            w1=paramWeights.narrow(-1, outRu1, 512),
            b1=paramBias.narrow(-1, outRu1, 512),
            w2=paramWeights.narrow(-1, outRu1+512, 512//4),
            b2=paramBias.narrow(-1, outRu1+512, 512//4),
            w3=paramWeights.narrow(-1, outRu1+512+512//4, 512//4),
            b3=paramBias.narrow(-1, outRu1+512+512//4, 512//4),
            w4=paramWeights.narrow(-1, outRu1+512+512//4+512//4, 512//4),
            b4=paramBias.narrow(-1, outRu1 + 512 + 512//4 + 512//4, 512//4))
        x = self.relu(x)
        outRu2 = outRu1 + 512 + 512//4 + 512//4 + 512//4

        x = torch.cat((x, layerUp4), dim=1)
        x = self.ResAda3(x)
        x = self.relu(x)
        x = self.ResUp3(
            x,
            w1=paramWeights.narrow(-1, outRu2, 256),
            b1=paramBias.narrow(-1, outRu2, 256),
            w2=paramWeights.narrow(-1, outRu2+256, 256//4),
            b2=paramBias.narrow(-1, outRu2+256, 256//4),
            w3=paramWeights.narrow(-1, outRu2+256+256//4, 256//4),
            b3=paramBias.narrow(-1, outRu2+256+256//4, 256//4),
            w4=paramWeights.narrow(-1, outRu2 + 256+256//4+256//4, 256//4),
            b4=paramBias.narrow(-1, outRu2 + 256 + 256//4 + 256//4, 256//4))
        x = self.relu(x)
        outRu3 = outRu2 + 256 + 256//4 + 256//4 + 256//4

        x = torch.cat((x, layerUp3), dim=1)
        x = self.ResAda4(x)
        x = self.relu(x)
        x = self.ResUp4(x,
                        w1=paramWeights.narrow(-1, outRu3, 128),
                        b1=paramBias.narrow(-1, outRu3, 128),
                        w2=paramWeights.narrow(-1, outRu3+128, 128),
                        b2=paramBias.narrow(-1, outRu3+128, 128),
                        w3=paramWeights.narrow(-1, outRu3+128+128, 128),
                        b3=paramBias.narrow(-1, outRu3+128+128, 128),
                        w4=paramWeights.narrow(-1, outRu3+128+128+128, 128),
                        b4=paramBias.narrow(-1, outRu3 + 128 + 128 + 128, 128))
        x = self.relu(x)
        outRu4 = outRu3 + 128 + 128 + 128 + 128

        # x = torch.cat((x, layerUp2), dim=1)
        # x = self.ResAda5(x)
        # x = self.relu(x)
        x = self.ResUp5(
            x,
            w1=paramWeights.narrow(-1, outRu4, 128),
            b1=paramBias.narrow(-1, outRu4, 128),
            w2=paramWeights.narrow(-1, outRu4+128, 128//4),
            b2=paramBias.narrow(-1, outRu4+128, 128//4),
            w3=paramWeights.narrow(-1, outRu4+128+128//4, 128//4),
            b3=paramBias.narrow(-1, outRu4+128+128//4, 128//4),
            w4=paramWeights.narrow(-1, outRu4 + 128+128//4+128//4, 128//4),
            b4=paramBias.narrow(-1, outRu4 + 128 + 128//4 + 128//4, 128//4))
        x = self.relu(x)
        outRu5 = outRu4 + 128 + 128//4 + 128//4 + 128//4
        # x = torch.cat((x, layerUp1), dim=1)
        # x = self.ResAda6(x)
        # x = self.relu(x)
        x = self.Res6(
            x,
            w1=paramWeights.narrow(-1, outRu5, 64),
            b1=paramBias.narrow(-1, outRu5, 64),
            w2=paramWeights.narrow(-1, outRu5+64, 64//4),
            b2=paramBias.narrow(-1, outRu5+64, 64//4),
            w3=paramWeights.narrow(-1, outRu5+64+64//4, 64//4),
            b3=paramBias.narrow(-1, outRu5+64+64//4, 64//4),
            w4=paramWeights.narrow(-1, outRu5+64+64//4+64//4, 64//4),
            b4=paramBias.narrow(-1, outRu5 + 64 + 64//4 + 64//4, 64//4))
        x = self.relu(x)
        x = self.Res7(x)
        x = self.sigmoid(x)
        # outRu6 = outRu5 + 64 + 64//4 + 64//4 + 64//4
        # print("PARAMMMMMM  : ", outRu6)
        return x


# ######################
#     Discriminator    #
# ######################
class Discriminator(nn.Module):
    """
    Class for the BigDiscriminator
    Architecture is almost the same as the embedder.

    Arguments:
        num_persons {int} -- The number of persons in the dataset. It's used to
        create the embeddings for each persons.
        Could be disabled in settings.py
        with the LOAD_EMBEDDINGS parameter to False.

    Returns:
        Create the model of the network (used then in utils.py -> load_models )
    """

    def __init__(self, num_persons, fine_tunning=False):
        """[summary]

        Arguments:
        num_persons {int} -- The number of persons in the dataset. It's used to
        Create the embeddings for each persons.
        Could be disabled in settings.py
        with the LOAD_EMBEDDINGS parameter to False.

        Keyword Arguments:
            fine_tunning {bool} -- will be used after... still not implemented
            (default: {False})
            Will be used to prevent the loading of embeddings to fintune only
            on one unknown person (variables are differents).
        """
        super(Discriminator, self).__init__()
        self.residual1 = ResidualBlockDown(6, 64)
        self.residual2 = ResidualBlockDown(64, 128)
        self.residual3 = ResidualBlock(128, 128)
        self.residual4 = ResidualBlockDown(128, 256)
        self.residual5 = ResidualBlock(256, 256)
        self.residual6 = ResidualBlockDown(256, 512)
        self.residual7 = ResidualBlock(512, 512)
        self.residual8 = ResidualBlockDown(512, 512)
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

        out = self.residual5(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual6(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual7(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = self.attention2(out)  # 2, 128, 56, 56
        out = self.relu(out)
        features_maps.append(out)

        out = self.residual8(out)  # 2, 512, 7,7
        out = self.relu(out)
        features_maps.append(out)

        out = torch.sum(out.view(out.size(0), out.size(1), -1), dim=2)  # b,512
        out = self.relu(out)
        final_out = self.fc(out)
        features_maps.append(out)

        w0 = self.w0.repeat(BATCH_SIZE).view(BATCH_SIZE, LATENT_SIZE)
        b = self.b.repeat(BATCH_SIZE)

        print("out : ", out.size())
        print("final_out : ", final_out.size())
        print("w0 : ", w0.size())
        print("b : ", b.size())
        print("self.embeddings(indexes) : ", self.embeddings(indexes).size())

        condition = torch.bmm(
            self.embeddings(indexes).view(-1, 1, LATENT_SIZE),
            (out+w0).view(BATCH_SIZE, LATENT_SIZE, 1)
        )
        final_out += condition.view(final_out.size())
        print("final_out : ", final_out.size())
        final_out = final_out.view(b.size())
        print("final_outvv : ", final_out.size())
        final_out += b
        return final_out, features_maps
