import inspect
import sys

import numpy as np
import torch
from torch import nn
from matplotlib.lines import Line2D
from models import Discriminator, Embedder, Generator
from settings import (PATH_WEIGHTS_DISCRIMINATOR, PATH_WEIGHTS_EMBEDDER,
                      PATH_WEIGHTS_GENERATOR,
                      DEVICE)

import matplotlib.style as mplstyle
import matplotlib.pyplot as plt

mplstyle.use(['dark_background', 'fast'])


def load_models(nb_pers, load_previous_state=True):
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
        discriminator.module.load_state_dict(
            torch.load(PATH_WEIGHTS_DISCRIMINATOR))
    # embedder = embedder.to(DEVICE)
    # generator = generator.to(DEVICE)
    # discriminator = discriminator.to(DEVICE)
    return embedder, generator, discriminator


def load_trained_models(nb_pers):
    embedder = Embedder()
    generator = Generator()
    discriminator = Discriminator(nb_pers)
    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)

    embedder.load_state_dict(torch.load(
        PATH_WEIGHTS_EMBEDDER, map_location="cuda"))
    generator.load_state_dict(torch.load(
        PATH_WEIGHTS_GENERATOR, map_location="cuda"))
    discriminator.load_state_dict(torch.load(
        PATH_WEIGHTS_DISCRIMINATOR, map_location="cuda"))

    discriminator = discriminator.to(DEVICE)
    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)

    embedder = embedder.eval()
    generator = generator.eval()
    discriminator = discriminator.eval()
    return embedder, generator, discriminator


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
                print('\n'+'-'*25+"\n| Poids disc sauvegardés |\n"+'-'*25+'\n')
                self.best_loss_Disc = loss
                torch.save(discriminator.module.state_dict(),
                           PATH_WEIGHTS_DISCRIMINATOR)
        else:
            if loss < self.best_loss_EmbGen:
                print('\n'+'-'*31+"\n| Poids Emb & Gen sauvegardés |\n"+'-'*31+'\n')
                self.best_loss_Emb = loss
                torch.save(embedder.module.state_dict(), PATH_WEIGHTS_EMBEDDER)
                torch.save(generator.module.state_dict(),
                           PATH_WEIGHTS_GENERATOR)

    def visualize(self,
                  gt_landmarks, synth_im, gt_im, *models,
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


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum((get_size(i, seen) for i in obj))
        except TypeError:
            print("0-D")

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen)
                    for s in obj.__slots__ if hasattr(obj, s))

    return size
