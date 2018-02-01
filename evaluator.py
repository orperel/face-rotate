import torch
from data_loader import UMDDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

use_cuda = False
g_dof = 45

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

def gen_one_hot(yaw, pitch, roll, dof=180, dof_quant=1):
    y = torch.LongTensor([yaw, pitch, roll])
    y_tensor = y.div_(dof_quant).type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], dof).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(-1)[None]
    if use_cuda:
        y_one_hot = y_one_hot.cuda()
    return y_one_hot

def show_random_samples():

    data_group_type = 'validation'
    data_group_zoom = 'debug'
    autoenc_model_path = os.path.join('models', 'last_autoencoder.pth')

    data = UMDDataset(path=os.path.join('dataset', data_group_type, data_group_zoom),
                      ypr_quant=True, deg_dim=g_dof, h_flip_augment=False, use_cuda=use_cuda)
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

    with torch.no_grad():
        autoenc = torch.load(autoenc_model_path)
        autoenc.eval()
        if not use_cuda:
            autoenc.cpu()
        else:
            autoenc.cuda()

        for batch in dataloader:

            x = Variable(batch['data'])
            y = Variable(batch['label'])

            fig = plt.figure()

            reconstructed_face = autoenc(x, y)
            show_img_var(x.data, fig, 1, 'Ground Truth')
            show_img_var(reconstructed_face[1].data, fig, 2, 'Reconstruction')
            altered_y = Variable(gen_one_hot(yaw=0//(180//g_dof), pitch=0, roll=0, dof=g_dof, dof_quant=180//g_dof))
            rotated_face = autoenc(x, altered_y)
            show_img_var(rotated_face[1].data, fig, 3, '(0,0,0) Rotation')

            plt.show()


def report_status_to_visdom(model, plot):
    plotter = torch.load(plot)
    plotter.plot_losses(window='Losses')
