import shutil
import os
import glob
import numpy as np
import torch
from torch import nn
from matplotlib.lines import Line2D
from models import Discriminator, Embedder, Generator

from bigmodels import Discriminator as BigDiscriminator
from bigmodels import Embedder as BigEmbedder
from bigmodels import Generator as BigGenerator

from settings import (PATH_WEIGHTS_DISCRIMINATOR, PATH_WEIGHTS_EMBEDDER,
                      PATH_WEIGHTS_GENERATOR, PATH_WEIGHTS_BIG_DISCRIMINATOR,
                      PATH_WEIGHTS_BIG_EMBEDDER, PATH_WEIGHTS_BIG_GENERATOR,
                      DEVICE, MODEL, LOAD_PREVIOUS, LOAD_EMBEDDINGS)

import matplotlib.style as mplstyle
import matplotlib.pyplot as plt

mplstyle.use(['dark_background', 'fast'])


def load_small_models(nb_pers, load_previous_state=LOAD_PREVIOUS,
                      load_embeddings=LOAD_EMBEDDINGS):
    embedder = Embedder()
    generator = Generator()
    discriminator = Discriminator(nb_pers)

    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    embedder = nn.DataParallel(
        embedder, device_ids=range(torch.cuda.device_count()))
    generator = nn.DataParallel(
        generator, device_ids=range(torch.cuda.device_count()))
    discriminator = nn.DataParallel(
        discriminator, device_ids=range(torch.cuda.device_count()))

    if load_previous_state:
        embedder.module.load_state_dict(torch.load(PATH_WEIGHTS_EMBEDDER))
        generator.module.load_state_dict(torch.load(PATH_WEIGHTS_GENERATOR))
        state_dict_discriminator = torch.load(PATH_WEIGHTS_DISCRIMINATOR)
        if load_embeddings:
            discriminator.load_state_dict(state_dict_discriminator)
        else:
            state_dict_discriminator.pop("embeddings.weight")
            discriminator.load_state_dict(state_dict_discriminator,
                                          strict=False)
    return embedder, generator, discriminator


def load_big_models(nb_pers, load_previous_state=LOAD_PREVIOUS,
                    load_embeddings=LOAD_EMBEDDINGS):
    embedder = BigEmbedder()
    generator = BigGenerator()
    discriminator = BigDiscriminator(nb_pers)

    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    embedder = nn.DataParallel(
        embedder, device_ids=range(torch.cuda.device_count()))
    generator = nn.DataParallel(
        generator, device_ids=range(torch.cuda.device_count()))
    discriminator = nn.DataParallel(
        discriminator, device_ids=range(torch.cuda.device_count()))

    if load_previous_state:
        embedder.module.load_state_dict(torch.load(PATH_WEIGHTS_BIG_EMBEDDER))
        generator.module.load_state_dict(
            torch.load(PATH_WEIGHTS_BIG_GENERATOR))
        state_dict_discriminator = torch.load(PATH_WEIGHTS_BIG_DISCRIMINATOR)
        if load_embeddings:
            discriminator.load_state_dict(state_dict_discriminator)
        else:
            state_dict_discriminator.pop("embeddings.weight")
            discriminator.load_state_dict(state_dict_discriminator,
                                          strict=False)
    return embedder, generator, discriminator


def load_models(nb_pers, load_previous_state=LOAD_PREVIOUS,
                load_embeddings=LOAD_EMBEDDINGS, model=MODEL):
    if model == "small":
        return load_small_models(nb_pers, load_previous_state, load_embeddings)
    elif model == "big":
        return load_big_models(nb_pers, load_previous_state, load_embeddings)


class Checkpoints:
    def __init__(self):
        self.losses = {"dsc": [], "cnt": [], "adv": [], "mch": []}
        self.best_loss_EmbGen = 1e10
        self.best_loss_Disc = 1e10

    def addCheckpoint(self, model, loss):
        loss = loss.detach()
        self.losses[model].append(loss)

    def save(self, model, loss, embedder, generator, discriminator):
        if model == "disc":
            if loss < self.best_loss_Disc:
                print('\n' + '-' * 25)
                print("| Poids disc sauvegardés |\n")
                print('-'*25)
                self.best_loss_Disc = loss
                if MODEL == 'small':
                    torch.save(discriminator.module.state_dict(),
                               PATH_WEIGHTS_DISCRIMINATOR)
                elif MODEL == "big":
                    torch.save(discriminator.module.state_dict(),
                               PATH_WEIGHTS_BIG_DISCRIMINATOR)
        else:
            if loss < self.best_loss_EmbGen:
                print('\n' + '-'*31)
                print("| Poids Emb & Gen sauvegardés |")
                print('-'*31)
                self.best_loss_Emb = loss
                if MODEL == 'small':
                    torch.save(embedder.module.state_dict(),
                               PATH_WEIGHTS_EMBEDDER)
                    torch.save(generator.module.state_dict(),
                               PATH_WEIGHTS_GENERATOR)
                elif MODEL == "big":
                    torch.save(embedder.module.state_dict(),
                               PATH_WEIGHTS_BIG_EMBEDDER)
                    torch.save(generator.module.state_dict(),
                               PATH_WEIGHTS_BIG_GENERATOR)

    def visualize(self, gt_landmarks, synth_im, gt_im, *models,
                  save_fig=False, name='plop', show=False):
        "-----------------------"
        # TODO Faire une vraie accuracy
        accuracy = 0.5
        "------------------------"
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), num='Mon')
        im_landmarks = gt_landmarks[0].detach().cpu().permute(1, 2, 0).numpy()
        im_synth = synth_im[0].detach().cpu().permute(1, 2, 0).numpy()
        im_gt = gt_im[0].detach().cpu().permute(1, 2, 0).numpy()

        axes[0, 0].clear()
        axes[0, 0].imshow(im_landmarks/im_landmarks.max())
        axes[0, 0].axis("off")
        axes[0, 0].set_title('Landmarks')

        axes[0, 1].clear()
        axes[0, 1].imshow(im_synth/im_synth.max())
        axes[0, 1].axis("off")
        axes[0, 1].set_title('Synthesized image')

        axes[0, 2].clear()
        axes[0, 2].imshow(im_gt/im_gt.max())
        axes[0, 2].axis("off")
        axes[0, 2].set_title('Ground truth')

        axes[1, 0].clear()
        axes[1, 0].plot(self.losses["dsc"], label='Disc loss')
        axes[1, 0].set_title('Disc loss')

        axes[1, 1].clear()
        axes[1, 1].plot(self.losses["adv"], label='Adv loss')
        axes[1, 1].plot(self.losses["mch"], label='Mch loss')
        axes[1, 1].plot(self.losses["cnt"], label='Cnt loss')
        axes[1, 1].set_title('EmbGen losses')
        axes[1, 1].legend()

        axes[1, 2].clear()
        axes[1, 2].plot(accuracy)
        axes[1, 2].set_title('Accuracy')

        for i, m in enumerate(models):
            ave_grads = []
            max_grads = []
            layers = []
            for n, p in m.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    layers.append('.'.join(n.split('.')[: -1]))
                    try:
                        gradient = p.grad.cpu().detach()
                        ave_grads.append(gradient.abs().mean())
                        max_grads.append(gradient.abs().max())
                    except AttributeError:
                        ave_grads.append(0)
                        max_grads.append(0)
            axes[2, i].clear()
            axes[2, i].bar(np.arange(len(max_grads)), max_grads,
                           alpha=0.5, lw=1, color="c")
            axes[2, i].bar(np.arange(len(ave_grads)), ave_grads,
                           alpha=0.7, lw=1, color="r")
            axes[2, i].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
            axes[2, i].set_xticks(np.arange(len(layers)))
            axes[2, i].set_xticklabels(layers, rotation="vertical",
                                       fontsize='small')
            axes[2, i].set_xlim(left=0, right=len(ave_grads))
            axes[2, i].set_ylim(bottom=0, top=max(ave_grads)+1)
            # zoom in on the lower gradient regions
            axes[2, i].set_xlabel("Layers")
            axes[2, i].set_ylabel("average gradient")
            axes[2, i].set_title(f"{m.__class__.__name__} gradient flow")
            axes[2, i].grid(True)
            axes[2, i].legend([Line2D([0], [0], color="c", lw=4),
                               Line2D([0], [0], color="r", lw=4)],
                              ['max-gradient', 'mean-gradient'])
        # if save_fig:
        #     fig.savefig(f"{ROOT_IMAGE}{name}.png", dpi=fig.dpi)
        # fig.canvas.draw_idle()
        # fig.canvas.flush_events()
        return fig


def plot_grad_flow(fig, axes, *models):
    '''
    Plots the gradients flowing through different layers in the net
    during training.
    Can be used for checking for possible gradient
    vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow'''
    for i, m in enumerate(models):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in m.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                p.detach()
                layers.append('.'.join(n.split('.')[:-1]))
                try:
                    ave_grads.append(p.grad.abs().mean())
                except AttributeError:
                    print("No gradient for layer : ", n)
                    ave_grads.append(0)
                try:
                    max_grads.append(p.grad.abs().max())
                except AttributeError:
                    max_grads.append(0)
        axes[i].bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1,
                    color="c")
        axes[i].bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1,
                    color="b")
        axes[i].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        axes[i].set_xticks(np.arange(len(layers)))
        axes[i].set_xticklabels(layers, rotation="vertical", fontsize='small')
        axes[i].set_xlim(left=0, right=len(ave_grads))
        axes[i].set_ylim(bottom=min(ave_grads), top=max(ave_grads))
        # zoom in on the lower gradient regions
        axes[i].set_xlabel("Layers")
        axes[i].set_ylabel("average gradient")
        axes[i].set_title(f"{m.__class__.__name__} gradient flow")
        axes[i].grid(True)
        axes[i].legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)],
                       ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig.cla()
    fig.clf()
    fig.canvas.draw()
    fig.canvas.flush_events()


def make_light_dataset(path_dataset, new_path):
    for folder in glob.glob(f"{new_path}/*"):
        shutil.rmtree(folder)

    for folder in glob.glob(f"{path_dataset}/*"):
        os.mkdir(f"{new_path}/{folder.split('/')[-1]}")

        for context in glob.glob(f"{folder}/*"):
            # print(context)
            nb_files = len(glob.glob(f"{context}/*"))
            if nb_files == 1:
                dest = f"{new_path}/{'/'.join(context.split('/')[-2:])}"
                shutil.copytree(context, dest)
                break
