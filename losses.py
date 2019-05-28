from torchvision import models,


class VggLoss(nn.module):
    def __init__(self, NB_CANNAUX, IMAGE_SIZE):
        super(VggLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True)

    def forward(self, image):
        return self.vgg(image)
