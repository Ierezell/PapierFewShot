import torch
from models_ldmk import Generator
from settings import DEVICE, BATCH_SIZE

FACTOR = 10

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
    input_noise.flatten()

    random_walk = torch.randn(len(input_noise), device=DEVICE) / FACTOR

    for i, value in enumerate(random_walk):
        input_noise[i] += value
        img = gen(input_noise.view(BATCH_SIZE, 10, 1, 1))
        if i % 10:
            img_list.append(img)
