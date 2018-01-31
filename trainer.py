import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader import UMDDataset
from plotter import Plotter
from model import FaderNetAutoencoder, FaderNetDiscriminator
from utils import clip_grad_norm
import os
import time


class FaderNetTrainer:

    def __init__(self, t_params):
        self.t_params = t_params
        self.use_cuda = t_params['use_cuda'] and torch.cuda.is_available()

        if t_params['ypr_quant']:
            attr_dim = t_params['deg_dim'] * 3
            self.adversarial_loss_func = nn.CrossEntropyLoss()
        else:
            attr_dim = 3
            self.adversarial_loss_func = nn.MSELoss()
            self.max_regress_loss = torch.FloatTensor(3).fill_(180 ** 2)
            if self.use_cuda:
                self.max_regress_loss = self.max_regress_loss.cuda()

        self.reconstruction_loss_func = nn.MSELoss()

        self.autoenc = FaderNetAutoencoder(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim)
        self.discrm = FaderNetDiscriminator(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim)
        self.autoenc_optimizer = optim.Adam(self.autoenc.parameters(),
                                            lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))
        self.discrm_optimizer = optim.Adam(self.discrm.parameters(),
                                           lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))

        self.ypr_quant = t_params['ypr_quant']
        self.total_iterations = 0
        self.plotter = Plotter(path=self.t_params['plot_path'])
        self.best_discrm_loss = float("inf")
        self.best_autoenc_loss = float("inf")

        self.lambda_e = t_params['autoenc_loss_reg_init']
        self.lambda_e_max = t_params['autoenc_loss_reg']
        self.lambda_e_step_size = (self.lambda_e_max - self.lambda_e) / t_params['autoenc_loss_reg_adaption_steps']
        self.gradient_max_norm = t_params['gradient_max_norm']

    def adversarial_loss(self, y, y_predict):
        loss = 0
        if self.ypr_quant:
            degs_dim = self.t_params['deg_dim']
            for angle_idx in range(0, 3*degs_dim, degs_dim):
                y_target = y[:, angle_idx:angle_idx+degs_dim].max(1)[1] # Index of target degree
                y_predict_target = y_predict[:, angle_idx:angle_idx+degs_dim]
                loss = loss + self.adversarial_loss_func(y_predict_target, y_target)
        else:
            loss = loss + self.adversarial_loss_func(y_predict, y)

        return loss

    def complementary_adversarial_loss(self, y, y_predict):
        loss = 0
        if self.ypr_quant:
            batch_size = y.size()[0]
            degs_dim = self.t_params['deg_dim']

            for angle_idx in range(0, 3*degs_dim, degs_dim):
                y_target = y[:, angle_idx:angle_idx+degs_dim].max(1)[1]  # Index of target degree
                delta = torch.LongTensor(batch_size).random_(degs_dim - 1) + 1
                if self.use_cuda:
                    delta = delta.cuda()
                y_target = (y_target + Variable(delta)) % degs_dim
                y_predict_target = y_predict[:, angle_idx:angle_idx+degs_dim]
                loss = loss + self.adversarial_loss_func(y_predict_target, y_target)
        else:
            loss = Variable(self.max_regress_loss) - self.adversarial_loss_func(y_predict, y)

        return loss

    def reconstruct_loss(self, x, x_reconstruct):
        return self.reconstruction_loss_func(x_reconstruct, x)

    def discr_iteration(self, batch, mode):

        if mode == 'Training':
            self.discrm.train()
            self.autoenc.eval()
        else:
            self.discrm.eval()
            self.autoenc.eval()

        x = Variable(batch['data'], requires_grad=False)
        y = Variable(batch['label'], requires_grad=False)

        with torch.no_grad():
            z = self.autoenc.encode(x)

        y_predict = self.discrm(z)

        loss = self.adversarial_loss(y, y_predict)

        if mode == 'Training':
            self.discrm_optimizer.zero_grad()
            loss.backward()  # Backprop
            if self.gradient_max_norm > 0:
                clip_grad_norm(self.discrm.parameters(), self.gradient_max_norm)
            self.discrm_optimizer.step()

        return loss

    def autoenc_iteration(self, batch, mode):

        if mode == 'Training':
            self.discrm.eval()
            self.autoenc.train()
        else:
            self.discrm.eval()
            self.autoenc.eval()

        x = Variable(batch['data'], requires_grad=False)
        y = Variable(batch['label'], requires_grad=False)

        z, x_reconstruct = self.autoenc(x, y)

        with torch.no_grad():
            y_predict = self.discrm(z)

        adversarial_loss = self.complementary_adversarial_loss(y, y_predict)
        reconstruction_loss = self.reconstruct_loss(x, x_reconstruct)
        loss = reconstruction_loss + self.lambda_e * adversarial_loss

        assert not (loss != loss).data.any(), "NaN result in loss function"

        if mode == 'Training':
            self.autoenc_optimizer.zero_grad()
            loss.backward()  # Backprop
            if self.gradient_max_norm > 0:
                clip_grad_norm(self.autoenc.parameters(), self.gradient_max_norm)
            self.autoenc_optimizer.step()

        return loss

    def step_single_epoch(self, t, dataloader, mode):

        start = time.time()
        d_mean_loss = 0
        ae_mean_loss = 0

        for batch in dataloader:
            discriminator_loss = self.discr_iteration(batch, mode)
            auto_encoder_loss = self.autoenc_iteration(batch, mode)

            self.lambda_e = min(self.lambda_e + self.lambda_e_step_size, self.lambda_e_max)
            d_mean_loss += discriminator_loss.data[0]  # Already averaged by #nn_outputs * #batch_size
            ae_mean_loss += auto_encoder_loss.data[0]

        d_mean_loss /= len(dataloader)  # Divide by number of samples
        ae_mean_loss /= len(dataloader)  # Divide by number of samples
        self.plotter.update_loss_plot_data(mode='Discriminator ' + mode, new_epoch=(t + 1), new_loss=d_mean_loss)
        self.plotter.update_loss_plot_data(mode='AutoEncoder ' + mode, new_epoch=(t + 1), new_loss=ae_mean_loss)

        end = time.time()
        print(mode + ' took ' + "{0:.2f}".format(end - start) + ' seconds')

        return d_mean_loss, ae_mean_loss

    def train(self):

        if self.use_cuda:
            self.discrm.cuda()
            self.autoenc.cuda()

        training_set_path = os.path.join(self.t_params['dataset_path'], 'training', self.t_params['data_group'])
        validation_set_path = os.path.join(self.t_params['dataset_path'], 'validation', self.t_params['data_group'])

        training_data = UMDDataset(path=training_set_path,
                                   ypr_quant=self.ypr_quant, deg_dim=self.t_params['deg_dim'],
                                   h_flip_augment=self.t_params['h_flip_augment'],
                                   use_cuda=self.use_cuda)
        validation_data = UMDDataset(path=validation_set_path, ypr_quant=self.ypr_quant,
                                     deg_dim=self.t_params['deg_dim'], h_flip_augment=self.t_params['h_flip_augment'],
                                     use_cuda=self.use_cuda)

        train_dataloader = DataLoader(training_data, batch_size=self.t_params['batch_size'],
                                      shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(validation_data, batch_size=1,
                                           shuffle=True, num_workers=0)

        if not os.path.exists(self.t_params['models_path']):
            os.makedirs(self.t_params['models_path'])

        for t in range(self.t_params['epochs']):

            self.total_iterations = t
            print('Starting epoch #' + str(t + 1))

            self.step_single_epoch(t=t, dataloader=train_dataloader, mode='Training')

            with torch.no_grad():
                d_mean_loss, ae_mean_loss = self.step_single_epoch(t=t, dataloader=validation_dataloader, mode='Validation')

            # Always save best model found in term of minimal loss
            if self.best_discrm_loss > d_mean_loss:
                self.best_discrm_loss = d_mean_loss
                torch.save(self.discrm, self.t_params['models_path'] + 'discriminator' + str(t+1) + '.pth')
            elif self.best_autoenc_loss > ae_mean_loss:
                self.best_autoenc_loss = ae_mean_loss
                torch.save(self.autoenc, self.t_params['models_path'] + 'autoencoder' + str(t+1) + '.pth')

            self.plotter.plot_losses(window='Loss')
            torch.save(self.discrm, self.t_params['models_path'] + 'last_discriminator.pth')
            torch.save(self.autoenc, self.t_params['models_path'] + 'last_autoencoder.pth')
            torch.save(self, self.t_params['models_path'] + 'last_trainer_state.pth')

        print('Training ended. Terminating gracefully..')
