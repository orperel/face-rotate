import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evaluator_params import evaluating_params
from sanity.gender_data_loader import UMDDataset


def denormalize_img(x):
    return x.add(1).div_(2)

def bgr_to_rgb(x):
    return np.fliplr(x.reshape(-1,3)).reshape(x.shape)

def show_img_var(x_tensor, fig, i, title):
    next_plt = fig.add_subplot(1, 3, i)
    next_plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                         labelleft='off')
    next_plt.set_title(title)

    x_tensor = denormalize_img(x_tensor)
    processed_img = np.rollaxis((x_tensor.cpu().numpy().squeeze()).transpose(), axis=1)
    plt.imshow(bgr_to_rgb(processed_img))

def gen_one_hot(male, use_cuda):
    y = torch.FloatTensor([male, not male]).view(1, -1)
    if use_cuda:
        y = y.cuda()
    return y

def show_gender_random_samples():

    use_cuda = evaluating_params['use_cuda']
    data_type = evaluating_params['data_type']
    data_group = evaluating_params['data_group']
    dof = evaluating_params['deg_dim']
    ypr_quant = evaluating_params['ypr_quant']
    ypr_regress = evaluating_params['ypr_regress']
    h_flip_aug = evaluating_params['h_flip_augment']

    turnToMale = True

    autoenc_model_path = os.path.join('models', 'last_autoencoder.pth')

    data = UMDDataset(path=os.path.join('dataset', data_type, data_group),
                      ypr_quant=ypr_quant, deg_dim=dof, ypr_regress=ypr_regress,
                      h_flip_augment=h_flip_aug, use_cuda=use_cuda)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        autoenc = torch.load(autoenc_model_path)
        autoenc.eval()
        autoenc.gpus_count = 1
        if not use_cuda:
            autoenc.cpu()
        else:
            autoenc.cuda()

        random = randint(0, 15)

        for idx, batch in enumerate(dataloader):

            if idx % 20 != random:
                continue

            x = Variable(batch['data'])
            y = Variable(batch['label'])

            if use_cuda:
                x = x.float().cuda()
                y = y.float().cuda()

            fig = plt.figure()

            reconstructed_face = autoenc(x, y)
            show_img_var(x.data, fig, 1, 'Ground Truth')
            show_img_var(reconstructed_face[1].data, fig, 2, 'Reconstruction')

            altered_y = gen_one_hot(male=turnToMale, use_cuda=use_cuda)

            male_face = autoenc(x, Variable(altered_y))

            altered_title = 'Apply "Male"' if turnToMale else 'Apply "Female"'
            show_img_var(male_face[1].data, fig, 3, altered_title)

            plt.show()


def report_gender_status_to_visdom(plot):
    plotter = torch.load(plot)
    plotter.plot_losses(window='Losses')
