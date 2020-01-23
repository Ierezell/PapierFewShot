from builtins import input
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from train_ldmk2 import Generator
from settings import DEVICE, BATCH_SIZE_LDMK
from preprocess import write_landmarks_on_image
plt.ion()
fig = plt.figure()
ax = fig.gca()
fig.show()

FACTOR = 2
NB_IMAGE = 25

if __name__ == '__main__':

    img_list = []

    gen = Generator()
    gen.eval()

    gen = gen.to(DEVICE)

    try:
        gen.load_state_dict(torch.load("./weights/ldmk_int/generator_224.pt",
                                       map_location=DEVICE))
    except RuntimeError:
        gen.load_state_dict(torch.load("./weights/ldmk_int/generator_224.bk",
                                       map_location=DEVICE))
    except FileNotFoundError:
        print("Weights not found")

    input_noise = torch.randn(BATCH_SIZE_LDMK, 100, device=DEVICE)

    # input_noise = Variable(Tensor(np.random.normal(0, 1, (ldmks.shape[0], 100))))

    for _ in range(NB_IMAGE):
        value = torch.randn(BATCH_SIZE_LDMK, input_noise.size(1),
                            device=DEVICE) / FACTOR
        input_noise += value
        img = gen(input_noise)

        gen_ldmk_im = []
        for k in range(img.size(0)):
            genli = np.zeros((224, 224, 3), np.float32)

            l_gen = img[k]*224

            genli = write_landmarks_on_image(genli, l_gen.int().cpu())
            genli = transforms.ToTensor()(genli)
            genli /= 255.0

            gen_ldmk_im.append(genli)

        gen_ldmk_im = torch.stack(gen_ldmk_im)
        img_list.append(gen_ldmk_im)

        plt.imshow(genli.permute(1, 2, 0))
        fig.canvas.draw()
        plt.pause(0.1)
    plt.close(fig)
    plt.ioff()
    grid = torchvision.utils.make_grid(torch.stack(img_list).view(-1, 3,
                                                                  224, 224),
                                       padding=4, nrow=5,
                                       normalize=True, scale_each=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
