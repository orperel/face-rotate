import torch
from data_loader import UMDDataset
from torch.utils.data import DataLoader
from evaluator_params import evaluating_params
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

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

def gen_one_hot(yaw, pitch, roll, use_cuda, dof=180, dof_quant=1):
    y = torch.LongTensor([yaw, pitch, roll])
    y_tensor = y.div_(dof_quant).type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], dof).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(-1)[None]
    if use_cuda:
        y_one_hot = y_one_hot.cuda()
    return y_one_hot

def show_random_samples():

    use_cuda = evaluating_params['use_cuda']
    data_type = evaluating_params['data_type']
    data_group = evaluating_params['data_group']
    dof = evaluating_params['deg_dim']
    ypr_quant = evaluating_params['ypr_quant']
    ypr_regress = evaluating_params['ypr_regress']
    h_flip_aug = evaluating_params['h_flip_augment']
    yaw = 45
    pitch = 0
    roll = 0

    autoenc_model_path = os.path.join('models', 'last_autoencoder.pth')

    data = UMDDataset(path=os.path.join('dataset', data_type, data_group),
                      ypr_quant=ypr_quant, deg_dim=dof, ypr_regress=ypr_regress,
                      h_flip_augment=h_flip_aug, use_cuda=use_cuda)
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

            if ypr_quant:
                altered_y_onehot = gen_one_hot(yaw=yaw, pitch=pitch, roll=roll, use_cuda=use_cuda,
                                               dof=dof, dof_quant=180//dof)
            if ypr_regress:
                altered_y_regress = torch.LongTensor([yaw, pitch, roll]).div_(180)[None]
                if use_cuda:
                    altered_y_regress = altered_y_regress.cuda()

            if ypr_quant and ypr_regress:
                altered_y = torch.cat((altered_y_regress.float(), altered_y_onehot), dim=1)
            elif ypr_quant:
                altered_y = altered_y_onehot
            elif ypr_regress:
                altered_y = altered_y_regress

            rotated_face = autoenc(x, Variable(altered_y))

            altered_title = '(' + str(yaw) + ', ' + str(pitch) + ', ' + str(roll) + ') Rotation'
            show_img_var(rotated_face[1].data, fig, 3, altered_title)

            plt.show()


def report_status_to_visdom(plot):
    plotter = torch.load(plot)
    plotter.plot_losses(window='Losses')
