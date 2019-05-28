import torch
from torchvision.models.vgg import vgg19
import torch.nn as nn


class vggLoss(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(vggLoss, self).__init__()
        self.vgg_layers = vgg19(pretrained=True).features
        self.layer_name_mapping = {
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
        loss = 0
        for name, module in self.vgg_layers._modules.items():
            gt = module(gt)
            synth = module(synth)
            if name in self.layer_name_mapping:
                # output_gt[self.layer_name_mapping[name]] = gt
                # output_synth[self.layer_name_mapping[name]] = synth
                loss += self.l1(gt, synth)

        return loss


x = torch.rand(2, 3, 224, 224)
y = torch.rand(2, 3, 224, 224)
loss = vggLoss()
print(loss(x, y))
