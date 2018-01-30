import torch
import torch.nn as nn
from data_loader import UMDDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np


def denormalize_img(x):
    return x.add(1).div_(2)

def show_img_var(x_tensor):
    x_tensor = denormalize_img(x_tensor)
    plt.imshow(np.rollaxis((x_tensor.cpu().numpy().squeeze()).transpose(), axis=1))

def gen_one_hot(yaw, pitch, roll, dof=180, dof_quant=1):
    y = torch.LongTensor([yaw, pitch, roll])
    y_tensor = y.div_(dof_quant).type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], dof).scatter_(1, y_tensor, 1)
    return y_one_hot.view(-1)[None].cuda()

def show_random_samples():

    data_group_type = 'validation'
    data_group_zoom = 'debug'
    autoenc_model_path = os.path.join('models', 'autoencoder10.pth')

    data = UMDDataset(path=os.path.join('dataset', data_group_type, data_group_zoom),
                      ypr_quant=True, deg_dim=180, use_cuda=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

    with torch.no_grad():
        autoenc = torch.load(autoenc_model_path)
        autoenc.eval()

        for batch in dataloader:

            x = Variable(batch['data'])
            y = Variable(batch['label'])

            reconstructed_face = autoenc(x, y)
            show_img_var(x.data)
            show_img_var(reconstructed_face[1].data)
            altered_y = Variable(gen_one_hot(yaw=45, pitch=0, roll=0))
            rotated_face = autoenc(x, altered_y)
            show_img_var(rotated_face[1].data)

show_random_samples()
