import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from settings import (ATTENTION, BATCH_SIZE, CONCAT,
                      DEVICE, HALF, LATENT_SIZE)
from utils import load_layers, Padding, adaIN
(ResidualBlock, ResidualBlockDown, ResidualBlockUp, Attention) = load_layers()

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
        super().__init__()
        self.padding = Padding(224)
        self.residual1 = ResidualBlockDown(6, 64)  # 64, 128
        self.residual2 = ResidualBlockDown(64, 128)  # 128, 64
        self.residual3 = ResidualBlockDown(128, 256)  # 256, 32
        self.residual4 = ResidualBlockDown(256, LATENT_SIZE)  # 512 , 16
        self.residual5 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 512 , 8
        self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 512 , 4

        # self.FcWeights = spectral_norm(nn.Linear(LATENT_SIZE, 8017))
        # self.FcBias = spectral_norm(nn.Linear(LATENT_SIZE, 8017))

        self.relu = nn.SELU()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        if ATTENTION:
            self.attention1 = Attention(256)
            self.attention2 = Attention(LATENT_SIZE)

    def forward(self, x):  # b, 12, 224, 224
        """Forward pass :

        The network should take a BATCH picture as input of size (B*3)*W*H
        It takes the pictures ONE by ONE to compute their latent representation
        and then take the mean of all this representation to get the batch one.
        Returns:
            Tensor -- Size 1*512 corresponding to the latent
                                        representation of this BATCH of image
        """
        x = self.padding(x)
        if CONCAT:
            layerUp1 = torch.zeros((BATCH_SIZE, 64, 112, 112),
                                   dtype=(torch.half if HALF else torch.float),
                                   device=DEVICE)
            layerUp2 = torch.zeros((BATCH_SIZE, 128, 56, 56),
                                   dtype=(torch.half if HALF else torch.float),
                                   device=DEVICE)
            layerUp3 = torch.zeros((BATCH_SIZE, 256, 28, 28),
                                   dtype=(torch.half if HALF else torch.float),
                                   device=DEVICE)
            layerUp4 = torch.zeros((BATCH_SIZE, LATENT_SIZE, 14, 14),
                                   dtype=(torch.half if HALF else torch.float),
                                   device=DEVICE)
        else:
            layerUp1, layerUp2, layerUp3, layerUp4 = None, None, None, None

        x = x.view(-1, 6, 256, 256)
        out = self.residual1(x)  # bxk, 64, 112, 112
        out = self.relu(out)
        if CONCAT:
            layerUp1 = torch.add(out, layerUp1)

        out = self.residual2(out)  # b, 128, 56, 56
        out = self.relu(out)
        if CONCAT:
            layerUp2 = torch.add(out, layerUp2)

        out = self.residual3(out)  # b, 128, 56, 56
        out = self.relu(out)
        if CONCAT:
            layerUp3 = torch.add(out, layerUp3)
        if ATTENTION:
            out = self.attention1(out)  # b, 128, 56, 56
            out = self.relu(out)

        out = self.residual4(out)  # b, 256, 28, 28
        out = self.relu(out)
        if CONCAT:
            layerUp4 = torch.add(out, layerUp4)

        out = self.residual5(out)  # b, 256, 28, 28
        out = self.relu(out)
        out = self.residual6(out)  # b, 256, 28, 28
        if ATTENTION:
            out = self.attention2(out)  # b, 128, 56, 56
            out = self.relu(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.relu(out)

        # paramWeights = self.relu(self.FcWeights(out)).squeeze()
        # paramBias = self.relu(self.FcBias(out)).squeeze()

        out = out.view(-1, x.size(1), LATENT_SIZE, 1)
        out = out.mean(dim=1)

        layersUp = (layerUp1, layerUp2, layerUp3, layerUp4)
        return out, layersUp


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
        super().__init__()
        NB_PARAM = 2*(512*2*5+512*2+512*2+512+256+256+128+128+64+64+3)
        self.padding = Padding(224)

        # Down
        self.ResDown1 = ResidualBlockDown(3, 64)  # 64, 112
        self.in1 = nn.InstanceNorm2d(64, affine=True)

        self.ResDown2 = ResidualBlockDown(64, 128)  # 128, 64
        self.in2 = nn.InstanceNorm2d(128, affine=True)

        self.ResDown3 = ResidualBlockDown(128, 256)  # 256, 32
        self.in3 = nn.InstanceNorm2d(256, affine=True)

        self.ResDown4 = ResidualBlockDown(256, LATENT_SIZE)  # 512, 16
        self.in4 = nn.InstanceNorm2d(LATENT_SIZE, affine=True)

        # Constant
        self.ResBlock_1 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16
        self.ResBlock_2 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16
        self.ResBlock_3 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16
        self.ResBlock_4 = ResidualBlock(LATENT_SIZE, LATENT_SIZE)  # 512, 16

        # Up
        self.ResUp1 = ResidualBlockUp(LATENT_SIZE, 256)   # 256, 32
        self.ResUp2 = ResidualBlockUp(256, 128)   # 128, 64
        self.ResUp3 = ResidualBlockUp(128, 64)  # 64, 112, 112
        self.ResUp4 = ResidualBlockUp(64, 3)  # 3, 224, 224
        # self.conv1 = spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
        # self.conv2 = spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
        self.relu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        if ATTENTION:
            self.attentionDown1 = Attention(256)
            self.attention1 = Attention(LATENT_SIZE)
            self.attentionUp1 = Attention(128)
            self.attentionUp2 = Attention(3)

        if CONCAT:
            self.Ada1 = spectral_norm(
                nn.Conv2d(64*2, 64, kernel_size=3, padding=1))
            self.Ada2 = spectral_norm(
                nn.Conv2d(128*2, 128, kernel_size=3, padding=1))
            self.Ada3 = spectral_norm(
                nn.Conv2d(256*2, 256, kernel_size=3, padding=1))
            self.Ada4 = spectral_norm(
                nn.Conv2d(LATENT_SIZE*2, LATENT_SIZE, kernel_size=3, padding=1))

        self.p = nn.Parameter(torch.rand(self.NB_PARAM,
                                         512).normal_(0.0, 0.02))
        self.psi = nn.Parameter(torch.rand(self.NB_PARAM, 1))

    def finetuning_init(self):
        if FINETUNNING:
            self.psi = nn.Parameter(
                torch.mm(self.p, self.e_finetuning.mean(dim=0)))

    def forward(self, img, weights, layersUp):
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
        layerUp1, layerUp2, layerUp3, layerUp4 = layersUp
        if FINETUNNING:
            e_psi = self.psi.unsqueeze(0)
            e_psi = e_psi.expand(weights.shape[0], self.NB_PARAM, 1)
        else:
            p = self.p.unsqueeze(0)
            p = p.expand(weights.shape[0], self.NB_PARAM, 512)
            e_psi = torch.bmm(p, weights)
        # ######
        # DOWN #
        # ######
        x = self.pad(img)

        x = self.ResDown1(x)
        x = self.in1(x)

        x = self.ResDown2(x)
        x = self.in2(x)

        x = self.ResDown3(x)
        x = self.in3(x)

        if ATTENTION:
            x = self.attentionDown1(x)
            x = self.relu(x)

        x = self.ResDown4(x)
        x = self.in4(x)

        # ##########
        # CONSTANT #
        # ##########
        x = self.ResBlock_1(x, mean=weights.narrow(-1, i, nb_params),
                            std=weights.narrow(-1, i, nb_params))
        x = self.ResBlock_2(x, mean=weights.narrow(-1, i, nb_params),
                            std=weights.narrow(-1, i, nb_params))
        x = self.ResBlock_3(x, mean=weights.narrow(-1, i, nb_params),
                            std=weights.narrow(-1, i, nb_params))

        if ATTENTION:
            x = self.attention1(x)
            x = self.relu(x)

        x = self.ResBlock_4(x, mean=weights.narrow(-1, i, nb_params),
                            std=weights.narrow(-1, i, nb_params))

        if CONCAT == "last":
            x = torch.cat((x, layerUp4), dim=1)
            x = self.Ada4(x)

        # ####
        # UP #
        # ####
        x = self.ResUp1(x, mean=weights.narrow(-1, i, nb_params),
                        std=weights.narrow(-1, i, nb_params))

        if CONCAT == "last":
            x = torch.cat((x, layerUp3), dim=1)
            x = self.Ada3(x)

        x = self.ResUp2(x, mean=weights.narrow(-1, i, nb_params),
                        std=weights.narrow(-1, i, nb_params))

        if CONCAT == "last":
            x = torch.cat((x, layerUp2), dim=1)
            x = self.Ada2(x)

        x = self.ResUp3(x, mean=weights.narrow(-1, i, nb_params),
                        std=weights.narrow(-1, i, nb_params))

        if ATTENTION:
            x = self.attentionUp1(x)
            x = self.relu(x)

        if CONCAT == "last":
            x = torch.cat((x, layerUp1), dim=1)
            x = self.Ada1(x)

        x = self.ResUp4(x, mean=weights.narrow(-1, i, nb_params),
                        std=weights.narrow(-1, i, nb_params))

        x = self.sig(x)
        x = x * 255
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
        super().__init__()
        self.padding = Padding(224)
        self.padding = Padding(224)
        self.residual1 = ResidualBlockDown(6, 64)  # 64, 128
        self.residual2 = ResidualBlockDown(64, 128)  # 128, 64
        self.residual3 = ResidualBlockDown(128, 256)  # 256, 32
        self.residual4 = ResidualBlockDown(256, LATENT_SIZE)  # 512 , 16
        self.residual5 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 512 , 8
        self.residual6 = ResidualBlockDown(LATENT_SIZE, LATENT_SIZE)  # 512 , 4
        self.embeddings = nn.Embedding(num_persons, LATENT_SIZE)
        self.w0 = nn.Parameter(torch.rand(LATENT_SIZE), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.relu = nn.SELU()
        self.fc = spectral_norm(nn.Linear(LATENT_SIZE, 1))
        self.tanh = nn.Tanh()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.W_i = nn.Parameter(torch.rand(512, 128))
        self.w_0 = nn.Parameter(torch.randn(512, 1))
        self.b = nn.Parameter(torch.randn(1))
        self.w_prime = nn.Parameter(torch.randn(512, 1))

    def finetuning_init(self):
        if self.finetuning:
            self.w_prime = nn.Parameter(
                self.w_0 + self.e_finetuning.mean(dim=0))

    def forward(self, x, ldmk, indexes):
        x = torch.cat((x, ldmk), dim=1)
        x = self.padding(x)
        out1 = self.residual1(x)  # bxk, 64, 112, 112
        out2 = self.residual2(out1)  # b, 128, 56, 56
        out3 = self.residual3(out2)  # b, 128, 56, 56
        if ATTENTION:
            out3 = self.attention1(out3)  # b, 128, 56, 56
            out3 = self.relu(out3)
        out4 = self.residual4(out3)  # b, 256, 28, 28
        out5 = self.residual5(out4)  # b, 256, 28, 28
        out6 = self.residual6(out5)  # b, 256, 28, 28
        out = self.pooling(out6)
        out = out.view(-1, 512, 1)
        final_out = self.fc(out)

        if self.finetuning:
            out = torch.bmm(out.transpose(1, 2), (self.w_prime.unsqueeze(
                0).expand(out.shape[0], 512, 1))) + self.b
        else:
            out = torch.bmm(out.transpose(
                1, 2), (self.W_i[:, indexes].unsqueeze(-1)).transpose(0, 1) + self.w_0) + self.b  # 1x1

        condition = torch.bmm(
            self.embeddings(indexes).view(-1, 1, LATENT_SIZE),
            (out+w0).view(BATCH_SIZE, LATENT_SIZE, 1)
        )
        final_out += condition.view(final_out.size())
        final_out = final_out.view(b.size())
        final_out += b
        final_out = self.tanh(final_out)
        return final_out
