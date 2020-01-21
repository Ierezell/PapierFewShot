import matplotlib as plt
import torch
import torchvision
from models_ldmk import Generator
from settings import DEVICE, BATCH_SIZE
FACTOR = 30
NB_IMAGE = 20

if __name__ == '__main__':

    img_list = []

    gen = Generator()
    gen = gen.to(DEVICE)

    try:
        gen.load_state_dict(torch.load("./weights/ldmk/gen.pt",
                                       map_location=DEVICE))
    except RuntimeError:
        gen.load_state_dict(torch.load("./weights/ldmk/gen.bk",
                                       map_location=DEVICE))
    except FileNotFoundError:
        print("Weights not found")

    input_noise = torch.randn(BATCH_SIZE, 10, 1, 1, device=DEVICE)

    for _ in range(NB_IMAGE):
        value = torch.randn(BATCH_SIZE, input_noise.size(1)) / FACTOR
        input_noise += value
        img = gen(input_noise)
        img_list.append(img)

    grid = torchvision.utils.make_grid(torch.stack(img_list), padding=4, nrow=2,
                                       normalize=True, scale_each=True)
    plt.imshow(grid.permute(1, 2, 0))
