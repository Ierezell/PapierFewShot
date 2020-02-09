import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19

from settings import BATCH_SIZE, DEVICE, LATENT_SIZE

"""
For the calculation of LCNT, we evaluate L1 loss between activations of
Conv1,6,11,20,29 VGG19 layers
and Conv1,6,11,18,25 VGGFace layers for real and
fake images.

We sum these losses with the weights equal to
1 · 10−2 for VGG19 and 2 · 10−3 for VGGFace terms
                      T           (i)      (i)
LFM(G, Dk) = E(s,x)  Sum 1/Ni [ ||D(s, x) − D(s, G(s))||],
                     i=1          k         k           1

"""


# from face_alignment import FaceAlignment, LandmarksType
# class ldmkLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.face_landmarks = FaceAlignment(LandmarksType._2D, device="cpu")

#     def forward(self, gt, synth):
#         score = 0
#         with torch.no_grad():
#             for i in range(gt.size(0)):
#                 ldmks_gt = self.face_landmarks.get_landmarks_from_image(
#                     gt[i].permute(2, 1, 0).data.cpu().numpy())
#                 ldmks_synth = self.face_landmarks.get_landmarks_from_image(
#                     synth[i].permute(2, 1, 0).data.cpu().numpy())
#                 print("ldmk  ", ldmks_gt)
#                 score += np.linalg.norm(ldmks_gt[0]-ldmks_synth[0])
#         return torch.from_numpy()
# #########
#  L_adv  #
# #########

class adverserialLoss(nn.Module):
    def __init__(self):
        super(adverserialLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, score_disc_synth, features_gt, features_synth):
        feature_loss = 0
        for ft_gt, ft_synth in zip(features_gt, features_synth):
            feature_loss += self.l1(ft_gt, ft_synth)
        feature_loss = feature_loss.expand_as(score_disc_synth)
        # loss /= len(features_synth)
        # loss /= 10.0
        loss_disc = -score_disc_synth
        return (loss_disc + feature_loss).sum(dim=0)

# #########
#  L_mch  #
# #########


class matchLoss(nn.Module):
    def __init__(self):
        super(matchLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, ei, Wi):
        ei = ei.view(BATCH_SIZE, LATENT_SIZE)
        Wi = Wi.view(BATCH_SIZE, LATENT_SIZE)
        return self.l1(ei, Wi)
        # return 80*(self.l1(ei, Wi)/BATCH_SIZE)


# #########
#  L_dsc  #
# #########
class discriminatorLoss(nn.Module):
    def __init__(self):
        super(discriminatorLoss, self).__init__()

    def forward(self, score_gt, score_synth):
        one = torch.ones_like(score_gt)
        zero = torch.zeros_like(score_gt)
        loss = torch.max(zero, one + score_synth) +\
            torch.max(zero, one - score_gt)
        return loss.sum(dim=0)


# class discriminatorLoss(nn.Module):
#     def __init__(self):
#         super(discriminatorLoss, self).__init__()

#     def forward(self, score_gt, score_synth):
#         one = torch.tensor((), device=DEVICE, dtype=torch.float)
#         eps = torch.tensor((),device=DEVICE,dtype=torch.float)
#         eps = eps.new_full(score_gt.size(), 1e-6)
#         one = one.new_ones(score_synth.size())+eps
#         loss = torch.log(score_gt+eps) + torch.log(one - score_synth)
#         loss /= score_gt.size(0)
#         return loss

# #########
#  L_cnt  #
# #########
class contentLoss(nn.Module):
    def __init__(self):
        super(contentLoss, self).__init__()
        self.vgg = vgg19(pretrained=True)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg_layers = self.vgg.features
        self.vgg_Face = vgg_face_dag(freeze=True)

        self.layer_name_mapping_vgg19 = {
            '1': "relu1",
            '6': "relu2",
            '11': "relu3",
            '20': "relu4",
            '29': "relu5",
        }

        self.layer_name_mapping_vggFace = {
            '1': "relu1_1",
            '6': "relu2_1",
            '11': "relu3_1",
            '18': "relu4_1",
            '25': "relu5_1",
        }

        # self.l1 = nn.MSELoss(reduction="mean")
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, gt, synth):
        # output_gt = {}
        # output_synth = {}
        gtVgg19 = gt.clone()
        synthVgg19 = synth.clone()
        gtVggFace = gt.clone()
        synthVggFace = synth.clone()

        lossVgg19 = torch.zeros(1, device=DEVICE)
        lossVggFace = torch.zeros(1, device=DEVICE)

        for name, module in self.vgg_layers._modules.items():
            with torch.no_grad():
                gtVgg19 = module(gtVgg19)
                synthVgg19 = module(synthVgg19)
            if name in self.layer_name_mapping_vgg19:
                lossVgg19 += self.l1(gtVgg19, synthVgg19)
                # If needed, output can be dictionaries of vgg feature for
                # each layer :
                # output_gt[self.layer_name_mapping[name]] = gt
                # output_synth[self.layer_name_mapping[name]] = synth
        for name, module in self.vgg_Face.named_children():
            gtVggFace = module(gtVggFace)
            synthVggFace = module(synthVggFace)
            if name in self.layer_name_mapping_vggFace.values():
                lossVggFace += self.l1(gtVggFace, synthVggFace)
            if name == "conv5_2":
                break
        return 25e-2 * lossVggFace + 15e-1*lossVgg19 + self.l1(gt, synth)


class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875,
                              104.76238250732422,
                              93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2],
                                  padding=0, dilation=1, ceil_mode=False)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2],
                                  padding=0, dilation=1, ceil_mode=False)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2],
                                  padding=0, dilation=1, ceil_mode=False)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0,
                                  dilation=1, ceil_mode=False)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3],
                                 stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2],
                                  padding=0, dilation=1, ceil_mode=False)

        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38


def vgg_face_dag(freeze=True, weights_path="./weights/vgg_face_dag.pth"):
    model = Vgg_face_dag()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.load_state_dict(torch.load(weights_path))
    return model
