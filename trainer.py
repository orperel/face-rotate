import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader import UMDDataset
from plotter import Plotter
from model import FaderNetAutoencoder, FaderNetDiscriminator
import os


class FaderNetTrainer:

    def __init__(self, t_params):
        self.t_params = t_params

        if t_params['ypr_quant']:
            attr_dim = 180 * 3
            self.adversarial_loss_func = nn.CrossEntropyLoss()
        else:
            attr_dim = 3
            self.adversarial_loss_func = nn.MSELoss()
            self.max_regress_loss = torch.LongTensor(3).fill_(180**2)

        self.reconstruction_loss_func = nn.MSELoss()
        self.autoenc = FaderNetAutoencoder(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim)
        self.discrm = FaderNetDiscriminator(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim)
        self.autoenc_optimizer = optim.Adam(self.autoenc.parameters(),
                                            lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))
        self.discrm_optimizer = optim.Adam(self.discrm.parameters(),
                                           lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))
        self.total_iterations = 0
        self.ypr_quant = t_params['ypr_quant']
        self.use_cuda = t_params['use_cuda'] and torch.cuda.is_available()

    def adversarial_loss(self, y, y_predict):
        loss = 0
        if self.ypr_quant:
            degs_dim = self.t_params['deg_dim']
            for angle_idx in range(0, 3*degs_dim, degs_dim):
                y_target = y[:, angle_idx:angle_idx+degs_dim].max(1)[1] # Index of target degree
                y_predict_target = y_predict[:, angle_idx:angle_idx+degs_dim]
                loss = loss + self.adversarial_loss_func(y_predict_target, Variable(y_target))
        else:
            loss = loss + self.adversarial_loss_func(Variable(y_predict), Variable(y))

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
                    delta.cuda()
                y_target = (y_target + Variable(delta)) % degs_dim
                y_predict_target = y_predict[:, angle_idx:angle_idx+degs_dim]
                loss = loss + self.adversarial_loss_func(y_predict_target, y_target)
        else:
            loss = loss + Variable(self.max_regress_loss) - self.adversarial_loss_func(Variable(y_predict), Variable(y))

        return loss

    def reconstruct_loss(self, x, x_reconstruct):
        return self.reconstruction_loss_func(x_reconstruct, x)

    def discr_iteration(self, batch):
        self.autoenc.eval()
        self.discrm.train()

        x = Variable(batch['data'], requires_grad=False)
        y = batch['label']

        if self.use_cuda:
            x.cuda()

        with torch.no_grad():
            z = self.autoenc.encode(x)

        y_predict = self.discrm(z)

        loss = self.adversarial_loss(y, y_predict)

        self.discrm_optimizer.zero_grad()
        loss.backward()  # Backprop
        self.discrm_optimizer.step()

        return loss

    def autoenc_iteration(self, batch):
        self.autoenc.train()
        self.discrm.eval()

        x = Variable(batch['data'], requires_grad=False)
        y = Variable(batch['label'])

        if self.use_cuda:
            x.cuda()
            y.cuda()

        z, x_reconstruct = self.autoenc(x, y)

        with torch.no_grad():
            y_predict = self.discrm(z)

        adversarial_loss = self.complementary_adversarial_loss(y, y_predict)
        reconstruction_loss = self.reconstruct_loss(x, x_reconstruct)
        loss = reconstruction_loss + self.t_params['autoenc_loss_reg'] * adversarial_loss

        assert not (loss != loss).data.any(), "NaN result in loss function"

        self.autoenc_optimizer.zero_grad()
        loss.backward()  # Backprop
        self.autoenc_optimizer.step()

        return loss

    def train(self):

        if self.use_cuda:
            self.discrm.cuda()
            self.autoenc.cuda()

        training_data = UMDDataset(path=os.path.join('dataset', 'training', self.t_params['data_group']),
                                   ypr_quant=self.ypr_quant,
                                   deg_dim=self.t_params['deg_dim'],
                                   use_cuda=self.use_cuda)
        train_dataloader = DataLoader(training_data, batch_size=self.t_params['batch_size'],
                                      shuffle=True, num_workers=0)

        validation_data = UMDDataset(path=os.path.join('dataset', 'validation', self.t_params['data_group']),
                                     ypr_quant = self.ypr_quant,
                                     deg_dim=self.t_params['deg_dim'],
                                     use_cuda=self.use_cuda)
        validation_dataloader = DataLoader(validation_data, batch_size=1,
                                           shuffle=True, num_workers=0)

        plotter = Plotter(path=self.t_params['plot_path'])
        best_loss = float("inf")

        for t in range(self.t_params['epochs']):

            self.total_iterations = t
            d_mean_loss = 0
            ae_mean_loss = 0
            print('Starting epoch #' + str(t + 1))

            for batch in train_dataloader:

                discriminator_loss = self.discr_iteration(batch)
                auto_encoder_loss = self.autoenc_iteration(batch)

                d_mean_loss += discriminator_loss.data[0]  # Already averaged by #nn_outputs * #batch_size
                ae_mean_loss += auto_encoder_loss.data[0]

            d_mean_loss /= len(train_dataloader) # Divide by number of training samples
            ae_mean_loss /= len(train_dataloader)  # Divide by number of training samples
            plotter.update_loss_plot_data(mode='Discriminator Training', new_epoch=(t+1), new_loss=d_mean_loss)
            plotter.update_loss_plot_data(mode='AutoEncoder Training', new_epoch=(t+1), new_loss=ae_mean_loss)

            vmean_loss = 0
            #
            # for vbatch in validation_dataloader:
            #     vx = Variable(vbatch['data'], volatile=True)
            #     vy = Variable(vbatch['label'], volatile=True)
            #     vpredict = model(vx)  # Feed forward
            #     vloss = criterion(vpredict, vy)  # Compute loss
            #     vmean_loss = vmean_loss + vloss.data[0]  # Already averaged by #nn_outputs * #batch_size
            #
            # vmean_loss /= len(validation_dataloader)  # Divide by number of validation samples

            # Always save best model found in term of minimal loss
            if best_loss > vmean_loss:
                best_loss = vmean_loss
                torch.save(self.discrm, './' + self.t_params['models_path'] + 'discriminator' + str(t+1) + '.pth')
                torch.save(self.autoenc, './' + self.t_params['models_path'] + 'autoencoder' + str(t+1) + '.pth')

            # plotter.update_loss_plot_data(mode='validation', new_epoch=(t + 1), new_loss=vmean_loss)

            plotter.plot_losses(window='Loss')