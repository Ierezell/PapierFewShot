import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from Archi import Discriminator, Embedder, Generator
from settings import (DEVICE, PATH_WEIGHTS_DISCRIMINATOR,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR,
                      ROOT_IMAGE)


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

    def visualize(self, ldm_gen, synth_im, gt_im, save_fig=False, name='plop'):
        "-----------------------"
        # TODO Faire une vraie accuracy
        accuracy = 0.5
        # TODO
        "------------------------"
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

        axes[1, 0].plot(self.loss_follow, label='Total loss')
        axes[1, 0].set_title('Total loss')

        axes[1, 1].plot(self.lossEmb, label='Emb loss')
        axes[1, 1].plot(self.lossDisc, label='Disc loss')
        axes[1, 1].set_title('Emb and disc losses')

        axes[1, 2].plot(accuracy)
        axes[1, 2].set_title('Accuracy')

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


def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net
    during training.
    Can be used for checking for possible gradient
    vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
