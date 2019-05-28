import torch
from matplotlib import pyplot as plt

from Archi import Discriminator, Embedder, Generator
from settings import (DEVICE, PATH_WEIGHTS_DISCRIMINATOR, ROOT_IMAGE
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR)


def load_models(load_previous_state=True):
    embedder = Embedder()
    generator = Generator()
    discriminator = Discriminator()
    if load_previous_state:
        embedder.load_state_dict(torch.load(PATH_WEIGHTS_EMBEDDER),
                                 map_location=torch.device(DEVICE))
        generator.load_state_dict(torch.load(PATH_WEIGHTS_GENERATOR),
                                  map_location=torch.device(DEVICE))
        discriminator.load_state_dict(torch.load(PATH_WEIGHTS_DISCRIMINATOR),
                                      map_location=torch.device(DEVICE))
    return embedder, generator, discriminator


class Checkpoints:
    def __init__(self):
        self.loss_follow = []
        self.lossEmbGen_follow = []
        self.lossDisc_follow = []
        self.losses_follow = []
        self.best_loss_EmbGen = 1e10
        self.best_loss_Disc = 1e10

    def addCheckpoint(self, lossEmbGen, lossDisc):
        self.loss_follow.append(lossEmbGen+lossDisc)
        self.lossEmbGen_follow.append(lossEmbGen)
        self.lossDisc_follow.append(lossDisc)
        self.losses_follow = [self.loss_follow, self.lossDisc_follow,
                              self.lossEmbGen_follow]

    def visualize(self, ldm_gen, synth_im, gt_im, loss, lossEmb, lossDisc,
                  save_fig=False, name='plop'):
        fig, axes = plt.subplots(2, 3)  # , figsize=(15, 10))
        axes[0, 0].imshow(ldm_gen.permute(1, 2, 0).cpu().detach().numpy())
        axes[0, 0].axis("off")
        axes[0, 0].set_title('Landmarks')

        axes[0, 1].imshow(synth_im.permute(1, 2, 0).cpu().detach().numpy())
        axes[0, 1].axis("off")
        axes[0, 1].set_title('Synthesized image')

        axes[0, 2].imshow(gt_im.permute(1, 2, 0).cpu().detach().numpy())
        axes[0, 2].axis("off")
        axes[0, 2].set_title('Ground truth')

        axes[1, 0].plot(loss)
        axes[1, 0].axis("off")
        axes[1, 0].set_title('Landmarks')

        axes[0, 1].imshow(synth_im.permute(1, 2, 0).cpu().detach().numpy())
        axes[0, 1].axis("off")
        axes[0, 1].set_title('Synthesized image')

        axes[0, 2].imshow(gt_im.permute(1, 2, 0).cpu().detach().numpy())
        axes[0, 2].axis("off")
        axes[0, 2].set_title('Ground truth')

        if save_fig:
            fig.savefig(f"{ROOT_IMAGE}{name}.png", dpi=fig.dpi)
        plt.show()

    def save(self, lossEmbGen, lossDisc, embedder, generator, discriminator):
        if lossDisc < self.best_loss_Disc:
            print('\n'+'-'*21+"\n| Poids sauvegardés |\n"+'-'*21+'\n')
            self.best_loss_Disc = lossDisc
            torch.save(discriminator.state_dict(), PATH_WEIGHTS_DISCRIMINATOR)

        if lossEmbGen < self.best_loss_EmbGen:
            print('\n'+'-'*21+"\n| Poids sauvegardés |\n"+'-'*21+'\n')
            self.best_loss_EmbGen = lossEmbGen
            torch.save(embedder.state_dict(), PATH_WEIGHTS_EMBEDDER)
            torch.save(generator.state_dict(), PATH_WEIGHTS_GENERATOR)
