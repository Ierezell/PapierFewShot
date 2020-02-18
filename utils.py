import glob
import os
import shutil
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib.lines import Line2D
from termcolor import colored
from tqdm import tqdm

from losses import adverserialLoss, contentLoss, discriminatorLoss, matchLoss
from RlModel import Policy
from settings import (DEVICE, HALF, LAYERS, LOAD_PREVIOUS, LOAD_PREVIOUS_RL,
                      MODEL, PARALLEL, PATH_WEIGHTS_DISCRIMINATOR,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR,
                      PATH_WEIGHTS_POLICY)

mplstyle.use(['dark_background', 'fast'])


def remove_bad_keys(model, path_checkpoint, resave=True):
    model_dict = model.state_dict()
    pre_model_dict = torch.load(path_checkpoint,  map_location=DEVICE)
    pre_model_dict = {k: v for k, v in pre_model_dict.items()
                      if k in model_dict}
    model_dict.update(pre_model_dict)
    if resave:
        torch.save(model_dict, path_checkpoint)
    return model_dict


def check_nan(x, msg=''):
    if torch.isnan(x).any():
        print(f"NAN IN {msg}{x[0]}")
    return torch.isnan(x).any()


def load_models(nb_pers, load_previous_state=LOAD_PREVIOUS, model=MODEL,
                root_path_weights=False, freeze=False):
    if root_path_weights:
        print(root_path_weights)
        path_discriminator = root_path_weights + "Discriminator.pt"
        path_embedder = root_path_weights + "Embedder.pt"
        path_generator = root_path_weights + "Generator.pt"
    else:
        print("Default")
        path_discriminator = PATH_WEIGHTS_DISCRIMINATOR
        path_embedder = PATH_WEIGHTS_EMBEDDER
        path_generator = PATH_WEIGHTS_GENERATOR

    if model == "small":
        from models2 import Discriminator, Embedder, Generator
        print("\tLoading Small Models (load previous)" if LOAD_PREVIOUS
              else "\tLoading Small Models (no load previous)")

        embedder = Embedder()
        generator = Generator()
        discriminator = Discriminator(nb_pers)

    elif model == "big":
        from bigmodels import BigDiscriminator, BigEmbedder, BigGenerator
        print("\tLoading Big Models (load previous)" if LOAD_PREVIOUS
              else "\tLoading Big Models (no load previous)")

        embedder = BigEmbedder()
        generator = BigGenerator()
        discriminator = BigDiscriminator(nb_pers)

    if HALF:
        embedder = embedder.half()
        generator = generator.half()
        discriminator = discriminator.half()

    if PARALLEL:
        embedder = nn.DataParallel(
            embedder, device_ids=range(torch.cuda.device_count()))
        generator = nn.DataParallel(
            generator, device_ids=range(torch.cuda.device_count()))
        discriminator = nn.DataParallel(
            discriminator, device_ids=range(torch.cuda.device_count()))

    if load_previous_state:
        # remove_bad_keys(embedder, path_embedder)
        # remove_bad_keys(embedder, path_embedder.replace(".pt", ".bk"))
        # remove_bad_keys(discriminator, path_discriminator)
        # remove_bad_keys(
        #     discriminator, path_discriminator.replace(".pt", ".bk"))
        # remove_bad_keys(generator, path_generator)
        # remove_bad_keys(generator, path_generator.replace(".pt", ".bk"))
        try:
            print(path_embedder)
            # input()
            if PARALLEL:
                embedder.module.load_state_dict(torch.load(path_embedder,
                                                           map_location=DEVICE))
            else:
                embedder.load_state_dict(torch.load(path_embedder))
        except RuntimeError:
            if PARALLEL:
                embedder.module.load_state_dict(
                    torch.load(path_embedder.replace(".pt", ".bk"),
                               map_location=DEVICE))
            else:
                embedder.load_state_dict(
                    torch.load(path_embedder.replace(".pt", ".bk"),
                               map_location=DEVICE))
        except FileNotFoundError:
            embedder.apply(weight_init)
            print(colored("\tFile not found, not loading weights embedder...", 'red'))
        try:
            if PARALLEL:
                generator.module.load_state_dict(
                    torch.load(path_generator,
                               map_location=DEVICE))
            else:
                generator.load_state_dict(
                    torch.load(path_generator,
                               map_location=DEVICE))

        except RuntimeError:
            if PARALLEL:
                generator.module.load_state_dict(
                    torch.load(path_generator.replace(".pt", ".bk"),
                               map_location=DEVICE))
            else:
                generator.load_state_dict(
                    torch.load(path_generator.replace(".pt", ".bk"),
                               map_location=DEVICE))
        except FileNotFoundError:
            generator.apply(weight_init)
            print(colored("\tFile not found, not loading weights generator...", 'red'))

        try:
            state_dict_discriminator = torch.load(path_discriminator,
                                                  map_location=DEVICE)
            weight_disc = True
        except RuntimeError:
            state_dict_discriminator = torch.load(
                path_discriminator.replace(".pt", ".bk"),
                map_location=DEVICE)
            weight_disc = True
        except FileNotFoundError:
            print(
                colored("\tFile not found, not loading weights discriminator...", 'red'))
            discriminator.apply(weight_init)
            weight_disc = False

        if weight_disc:
            try:
                if PARALLEL:
                    discriminator.module.load_state_dict(
                        state_dict_discriminator)
                else:
                    discriminator.load_state_dict(
                        state_dict_discriminator)
            except RuntimeError:
                print(colored("Wrong dataset, different number of persons", 'red'))
                print(colored("Loading disc without embeddings ", 'red'))
                state_dict_discriminator.pop("embeddings.weight")
                if PARALLEL:
                    discriminator.module.load_state_dict(
                        state_dict_discriminator, strict=False)
                else:
                    discriminator.load_state_dict(state_dict_discriminator,
                                                  strict=False)
    else:
        embedder.apply(weight_init)
        generator.apply(weight_init)
        discriminator.apply(weight_init)

    embedder = embedder.to(DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    if freeze:
        embedder = embedder.eval()
        discriminator = discriminator.eval()
        generator = generator.eval()
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in generator.parameters():
            param.requires_grad = False
        for param in embedder.parameters():
            param.requires_grad = False
    return embedder, generator, discriminator


def load_rl_model(load_previous_state=LOAD_PREVIOUS_RL):
    policy = Policy()
    policy = policy.to(DEVICE)
    policy = nn.DataParallel(
        policy, device_ids=range(torch.cuda.device_count()))

    if load_previous_state:
        print("Loading Rl Policy (pretrained)" if LOAD_PREVIOUS_RL
              else "Loading Rl Policy (no pretrained)")
        policy.module.load_state_dict(torch.load(PATH_WEIGHTS_POLICY))
    return policy


def load_losses():
    advLoss = adverserialLoss()
    mchLoss = matchLoss()
    cntLoss = contentLoss()
    dscLoss = discriminatorLoss()

    if HALF:
        advLoss = advLoss.half()
        mchLoss = mchLoss.half()
        cntLoss = cntLoss.half()
        dscLoss = dscLoss.half()

    if PARALLEL:
        advLoss = nn.DataParallel(
            advLoss, device_ids=range(torch.cuda.device_count()))
        mchLoss = nn.DataParallel(
            mchLoss, device_ids=range(torch.cuda.device_count()))
        cntLoss = nn.DataParallel(
            cntLoss, device_ids=range(torch.cuda.device_count()))
        dscLoss = nn.DataParallel(
            dscLoss, device_ids=range(torch.cuda.device_count()))

    advLoss = advLoss.to(DEVICE)
    mchLoss = mchLoss.to(DEVICE)
    cntLoss = cntLoss.to(DEVICE)
    dscLoss = dscLoss.to(DEVICE)

    return advLoss, mchLoss, cntLoss, dscLoss


def load_layers(size=LAYERS):
    if size == "small":
        from layers import (Attention, ResidualBlock,
                            ResidualBlockDown, ResidualBlockUp)
        print("\tLoading small layers")
        return ResidualBlock, ResidualBlockDown, ResidualBlockUp, Attention
    elif size == "big":
        from biglayers import (BigResidualBlock, BigResidualBlockDown,
                               BigResidualBlockUp, Attention)
        print("\tLoading big layers")
        return (BigResidualBlock, BigResidualBlockDown,
                BigResidualBlockUp, Attention)


class CheckpointsFewShots:
    def __init__(self, len_loader):
        self.best_loss_EmbGen = 1000
        self.best_loss_Disc = 1000
        self.step_disc = 0
        self.step_EmbGen = 0
        self.save_every = len_loader//4

    def save(self, model, loss, embedder, generator, discriminator,
             path_emb=PATH_WEIGHTS_EMBEDDER, path_gen=PATH_WEIGHTS_GENERATOR,
             path_disc=PATH_WEIGHTS_DISCRIMINATOR):

        loss = loss.detach()

        if model == "disc":
            self.step_disc += 1
            if loss < self.best_loss_Disc or self.step_disc > self.save_every:
                self.step_disc = 0
                tqdm.write(colored('\n' + '-'*25 + '\n' +
                                   "| Poids disc sauvegardes |" + '\n' + '-'*25,
                                   'green'))
                self.best_loss_Disc = loss
                if PARALLEL:
                    torch.save(discriminator.module.state_dict(),
                               path_disc.replace(".pt",
                                                 ".bk"))
                else:
                    torch.save(discriminator.state_dict(),
                               path_disc.replace(".pt", ".bk"))
                copyfile(path_disc.replace(".pt", ".bk"), path_disc)
        else:
            self.step_EmbGen += 1
            if (loss < self.best_loss_EmbGen or
                    self.step_EmbGen > self.save_every):
                self.step_EmbGen = 0
                tqdm.write(colored("\n" + "-" * 31 + '\n' +
                                   "| Poids Emb & Gen sauvegardes |" + '\n' + "-"*31, 'green'))
                self.best_loss_EmbGen = loss
                if PARALLEL:
                    torch.save(embedder.module.state_dict(),
                               path_emb.replace(".pt", ".bk"))
                    torch.save(generator.module.state_dict(),
                               path_gen.replace(".pt", ".bk"))
                else:
                    torch.save(embedder.state_dict(),
                               path_emb.replace(".pt", ".bk"))
                    torch.save(generator.state_dict(),
                               path_gen.replace(".pt", ".bk"))
                copyfile(path_emb.replace(".pt", ".bk"), path_emb)
                copyfile(path_gen.replace(".pt", ".bk"), path_gen)


class CheckpointsRl:
    def __init__(self):
        self.losses = {"Rl": []}
        self.best_loss = 1e10
        self.last_save = 0

    def addCheckpoint(self, model, loss):
        loss = loss.detach()
        self.losses[model].append(loss)
        self.last_save += 1

    def save(self, loss, policy):
        if loss < self.best_loss or self.last_save > 100:
            self.last_save = 0
            print('\n' + '-'*20)
            print("| Poids sauvegardes |")
            print('-'*20)
            self.best_loss = loss
            torch.save(policy.module.state_dict(),  PATH_WEIGHTS_POLICY)


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


def print_parameters(model):
    trainParamModel = sum([np.prod(p.size()) if p.requires_grad else 0
                           for p in model.parameters()])
    try:
        name = model.module.__class__.__name__
    except AttributeError:
        name = model.__class__.__name__
    print(f"Nombre de parametres {name}: {trainParamModel:,}")


def print_device(model):
    try:
        try:
            name = model.module.__class__.__name__
            device_param = next(model.module.parameters()).device
        except AttributeError:
            name = model.__class__.__name__
            device_param = next(model.parameters()).device

        print(f"{name} est sur {device_param}")
    except StopIteration:
        print(f"{name} n'as pas de parametres")


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Padding(nn.Module):
    def __init__(self, in_shape):
        super(Padding, self).__init__()

        self.zero_pad = nn.ZeroPad2d(self.findPadSize(in_shape))

    def forward(self, x):
        out = self.zero_pad(x)
        return out

    def findPadSize(self, in_shape):
        if in_shape < 256:
            pad_size = (256 - in_shape)//2
        else:
            pad_size = 0
        return pad_size


def adaIN(feature, mean_style, std_style, eps=1e-5):
    B, C, H, W = feature.shape

    feature = feature.view(B, C, -1)

    std_feat = (torch.std(feature, dim=2) + eps).view(B, C, 1)
    mean_feat = torch.mean(feature, dim=2).view(B, C, 1)

    adain = std_style * (feature - mean_feat)/std_feat + mean_style

    adain = adain.view(B, C, H, W)
    return adain
