from torchvision.models.vgg import vgg19
import torch.nn as nn
from Archi import Vgg_face
import torch
from settings import ROOT_WEIGHTS
"""

For the calculation of LCNT, we evaluate L1 loss between activations of
Conv1,6,11,20,29 VGG19 layers
and Conv1,6,11,18,25 VGGFace layers for real and
fake images.

We sum these losses with the weights equal to
1 · 10−2 for VGG19 and 2 · 10−3 for VGGFace terms
"""


class vggLoss(nn.Module):
    def __init__(self):
        super(vggLoss, self).__init__()
        self.vgg_layers = vgg19(pretrained=True).features

        self.vgg_Face = Vgg_face()
        self.vgg_Face.load_state_dict(
            torch.load(f'{ROOT_WEIGHTS}vgg_face.pth')
        )

        self.vgg_layers_Face = self.vgg_Face.features
        self.layer_name_mapping_vgg19 = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        self.layer_name_mapping_vggFace = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        self.l1 = nn.L1Loss()

    def forward(self, gt, synth):
        # output_gt = {}
        # output_synth = {}
        gtFace = gt.copy()
        synthFace = synth.copy()
        loss = 0
        for name, module in self.vgg_layers._modules.items():
            gt = module(gt)
            synth = module(synth)
            if name in self.layer_name_mapping_vgg19:
                # If needed, output can be dictionaries of vgg feature for each
                # layer :

                # output_gt[self.layer_name_mapping[name]] = gt
                # output_synth[self.layer_name_mapping[name]] = synth
                loss += 1e-2*self.l1(gt, synth)

        for name, module in self.vgg_layers_Face._modules.items():
            gtFace = module(gtFace)
            synthFace = module(synthFace)
            if name in self.layer_name_mapping_vgg19:
                loss += 2e-3*self.l1(gtFace, synth)
        return loss
