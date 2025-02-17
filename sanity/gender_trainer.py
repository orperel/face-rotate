import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F

from model import FaderNetAutoencoder, FaderNetDiscriminator
from plotter import Plotter
from sanity.gender_data_loader import UMDDataset, SubGroupsRandomSampler
from utils import query_available_gpus


class GenderFaderNetTrainer:

    def __init__(self, t_params):
        self.t_params = t_params

        self.use_cuda = t_params['use_cuda'] and torch.cuda.is_available()
        self.gpus_count = query_available_gpus() if self.use_cuda else 0
        if t_params['force-gpu-count'] > 0:
            self.gpus_count = t_params['force-gpu-count']
        if self.gpus_count == 0:
            self.use_cuda = False
        elif self.gpus_count > 1 and self.gpus_count > t_params['default-gpu']:
            torch.cuda.set_device(t_params['default-gpu'])

        attr_dim = 2
        self.adversarial_loss_func = nn.CrossEntropyLoss()
        self.reconstruction_loss_func = nn.MSELoss()

        self.autoenc = FaderNetAutoencoder(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim,
                                           stn_transform=t_params['stn'],
                                           gpus_count=self.gpus_count)
        self.discrm = FaderNetDiscriminator(num_of_layers=t_params['autoenc_layer_count'], attr_dim=attr_dim,
                                            gpus_count=self.gpus_count)
        fader_ae = torch.load('fader_init_ae.pth')
        fader_dis = torch.load('fader_init_dis.pth')

        new = list(fader_ae.state_dict().items())
        my_model_kvpair = self.autoenc.state_dict()
        count = 0
        for key, value in my_model_kvpair.items():
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1

        new = list(fader_dis.state_dict().items())
        my_model_kvpair = self.discrm.state_dict()
        count = 0
        for key, value in my_model_kvpair.items():
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1

        logging.debug(self.autoenc)
        logging.debug(self.discrm)
        self.autoenc_optimizer = optim.Adam(self.autoenc.parameters(),
                                            lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))
        self.discrm_optimizer = optim.Adam(self.discrm.parameters(),
                                           lr=t_params['learning_rate'], betas=(t_params['beta1'], t_params['beta2']))

        self.ypr_quant = t_params['ypr_quant']
        self.ypr_regress = t_params['ypr_regress']
        self.total_epochs = 0
        self.plotter = Plotter(path=self.t_params['plot_path'])
        self.best_discrm_loss = float("inf")
        self.best_autoenc_loss = float("inf")

        if self.t_params['training_samples_per_epoch'] == 0:
            self.t_params['training_samples_per_epoch'] = 2**32
        if self.t_params['validation_samples_per_epoch'] == 0:
            self.t_params['validation_samples_per_epoch'] = 2**32

        self.lambda_e = t_params['autoenc_loss_reg_init']
        self.lambda_e_max = t_params['autoenc_loss_reg']
        self.lambda_e_step_size = (self.lambda_e_max - self.lambda_e) / t_params['autoenc_loss_reg_adaption_steps']
        self.gradient_max_norm = t_params['gradient_max_norm']

    def get_attr_loss(self, preds, orig, flip):
        """
        Compute attributes loss.
        """
        assert type(flip) is bool
        k = 0
        loss = 0
        # categorical
        x = preds[:,:].contiguous()
        y = orig[:, :].max(1)[1].view(-1)
        if flip:
            # generate different categories
            shift = torch.LongTensor(y.size()).random_(2 - 1) + 1
            y = (y + Variable(shift.cuda())) % 2
        loss += F.cross_entropy(x, y)
        k += 2
        return loss

    def adversarial_loss(self, y, y_predict):
        loss = self.get_attr_loss(y_predict, y, False)
        # y_target = y.max(1)[1] # Index of target degree
        # y_predict_target = y_predict
        # loss = self.adversarial_loss_func(y_predict_target, y_target)
        return loss

    def my_complementary_adversarial_loss(self, y, y_predict):
        batch_size = y.size()[0]
        degs_dim = self.t_params['deg_dim']

        y_target = y.max(1)[1]  # Index of target degree
        delta = torch.LongTensor(batch_size).random_(degs_dim - 1) + 1
        if self.use_cuda:
            delta = delta.cuda()
        y_target = (y_target + Variable(delta)) % degs_dim
        y_predict_target = y_predict
        loss = self.adversarial_loss_func(y_predict_target, y_target)

        return loss

    def complementary_adversarial_loss(self, y, y_predict):
        loss = self.get_attr_loss(y_predict, y, True)
        #orig_loss = self.my_complementary_adversarial_loss(y, y_predict)
        return loss


    def reconstruct_loss(self, x, x_reconstruct):
        return ((x - x_reconstruct) ** 2).mean()
        # return self.reconstruction_loss_func(x_reconstruct, x)

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
            z = self.autoenc.encode(Variable(x.data))

        y_predict = self.discrm(Variable(z.data))
        logging.debug('Discriminator predict: ' + str(y_predict))

        loss = self.adversarial_loss(y, y_predict)
        logging.debug('Discriminator adversarial loss: ' + str(loss))

        assert not (loss != loss).data.any(), "NaN result in loss function"

        if mode == 'Training':
            self.discrm_optimizer.zero_grad()
            loss.backward()  # Backprop
            if self.gradient_max_norm > 0:
                nn.utils.clip_grad_norm(self.discrm.parameters(), self.gradient_max_norm)
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
        reconstruction_loss = self.reconstruct_loss(x, x_reconstruct)
        logging.debug('AutoEnc reconstruction loss: ' + str(reconstruction_loss))

        y_predict = self.discrm(z)
        logging.debug('AutoEnc adv predict: ' + str(y_predict))

        adversarial_loss = self.complementary_adversarial_loss(y, y_predict)
        logging.debug('AutoEnc comp adv loss: ' + str(adversarial_loss))

        loss = reconstruction_loss + self.lambda_e * adversarial_loss
        assert not (loss != loss).data.any(), "NaN result in loss function"

        if mode == 'Training':
            self.autoenc_optimizer.zero_grad()
            loss.backward()  # Backprop
            if self.gradient_max_norm > 0:
                nn.utils.clip_grad_norm(self.autoenc.parameters(), self.gradient_max_norm)
            self.autoenc_optimizer.step()

        return loss

    def step_single_epoch(self, t, dataloader, mode, max_samples):

        start = time.time()
        total_iterations = 0
        iterations_for_log = 0
        d_mean_loss = 0
        ae_mean_loss = 0
        batch_size = self.t_params['batch_size']

        turns_pattern = ['D', 'AE']
        pattern_idx = 0

        for batch in dataloader:

            if self.use_cuda:
                batch = {'data': batch['data'].cuda(async=True), 'label': batch['label'].cuda(async=True)}

            if turns_pattern[pattern_idx] == 'D':
                discriminator_loss = self.discr_iteration(batch, mode)
                d_mean_loss += discriminator_loss.data[0]  # Already averaged by #nn_outputs * #batch_size
                logging.info('Discriminator total loss: ' + "{0:.5f}".format(discriminator_loss.data[0]))
            elif turns_pattern[pattern_idx] == 'AE':
                auto_encoder_loss = self.autoenc_iteration(batch, mode)
                if mode == 'Training':
                    self.lambda_e = min(self.lambda_e + self.lambda_e_step_size, self.lambda_e_max)
                ae_mean_loss += auto_encoder_loss.data[0]
                logging.info('AutoEncoder total loss: ' + "{0:.5f}".format(auto_encoder_loss.data[0]))

            pattern_idx = (pattern_idx + 1) % len(turns_pattern)
            if pattern_idx == 0:
                total_iterations += batch_size
                iterations_for_log += 1
                if total_iterations % batch_size * 5 == 0:
                    logging.info('Processed %i iterations', iterations_for_log)
                if total_iterations >= max_samples:
                    break

        processed_samples_count = total_iterations
        d_mean_loss /= processed_samples_count  # Divide by number of samples
        ae_mean_loss /= processed_samples_count  # Divide by number of samples
        logging.info('Epoch mean loss: [AutoEnc: ' + "{0:.5f}".format(ae_mean_loss) +
                     ' Discriminator: ' + "{0:.5f}".format(d_mean_loss) + ']')
        self.plotter.update_loss_plot_data(network='Discriminator', mode=mode, new_epoch=(t + 1), new_loss=d_mean_loss)
        self.plotter.update_loss_plot_data(network='AutoEncoder', mode=mode, new_epoch=(t + 1), new_loss=ae_mean_loss)

        end = time.time()
        logging.info(mode + ' took ' + "{0:.3f}".format(end - start) + ' seconds')

        return d_mean_loss, ae_mean_loss

    def train(self):

        if self.gpus_count == 0:
            logging.info('Starting training on CPU..')
        else:
            logging.info('Starting training on %i GPU(s)' % (self.gpus_count))

        if self.use_cuda:
            self.discrm = self.discrm.cuda()
            self.autoenc = self.autoenc.cuda()
        else:
            self.discrm = self.discrm.cpu()
            self.autoenc = self.autoenc.cpu()

        torch.backends.cudnn.benchmark = True

        training_set_path = os.path.join(self.t_params['dataset_path'], 'training', self.t_params['data_group'])
        validation_set_path = os.path.join(self.t_params['dataset_path'], 'validation', self.t_params['data_group'])

        training_data = UMDDataset(path=training_set_path,
                                   ypr_quant=self.ypr_quant, deg_dim=self.t_params['deg_dim'],
                                   ypr_regress=self.ypr_regress,
                                   h_flip_augment=self.t_params['h_flip_augment'],
                                   use_cuda=self.use_cuda)
        logging.info(str(len(training_data)) + ' training samples loaded.')
        validation_data = UMDDataset(path=validation_set_path, ypr_quant=self.ypr_quant, deg_dim=self.t_params['deg_dim'],
                                     ypr_regress=self.ypr_regress, h_flip_augment=self.t_params['h_flip_augment'],
                                     use_cuda=self.use_cuda)
        logging.info(str(len(validation_data)) + ' validation samples loaded.')

        train_dataloader = DataLoader(training_data, batch_size=self.t_params['batch_size'],
                                      shuffle=False, sampler=SubGroupsRandomSampler(training_data), num_workers=1,
                                      pin_memory=True)
        validation_dataloader = DataLoader(validation_data, batch_size=1,
                                           shuffle=False, sampler=SubGroupsRandomSampler(validation_data), num_workers=1,
                                           pin_memory=True)

        if not os.path.exists(self.t_params['models_path']):
            os.makedirs(self.t_params['models_path'])

        for t in range(self.total_epochs, self.t_params['epochs']):
            self.total_epochs = t
            logging.info('Starting epoch #' + str(t + 1))

            self.step_single_epoch(t=t, dataloader=train_dataloader, mode='Training',
                                   max_samples=self.t_params['training_samples_per_epoch'])

            with torch.no_grad():
                d_mean_loss, ae_mean_loss = self.step_single_epoch(t=t, dataloader=validation_dataloader, mode='Validation',
                                                                   max_samples=self.t_params['training_samples_per_epoch'])

            # Always save best model found in term of minimal loss
            if self.best_discrm_loss > d_mean_loss:
                self.best_discrm_loss = d_mean_loss
                torch.save(self.discrm,
                           self.t_params['models_path'] + 'best_d_loss_discriminator' + str(t + 1) + '.pth')
                torch.save(self.autoenc,
                           self.t_params['models_path'] + 'best_d_loss_autoencoder' + str(t + 1) + '.pth')
            elif self.best_autoenc_loss > ae_mean_loss:
                self.best_autoenc_loss = ae_mean_loss
                torch.save(self.discrm,
                           self.t_params['models_path'] + 'best_ae_loss_discriminator' + str(t + 1) + '.pth')
                torch.save(self.autoenc,
                           self.t_params['models_path'] + 'best_ae_loss_autoencoder' + str(t + 1) + '.pth')

            # Save periodically
            if t % 5 == 0:
                torch.save(self.discrm, self.t_params['models_path'] + 'discriminator' + str(t + 1) + '.pth')
                torch.save(self.autoenc, self.t_params['models_path'] + 'autoencoder' + str(t + 1) + '.pth')

            self.plotter.plot_losses(window='Losses')
            torch.save(self.discrm, self.t_params['models_path'] + 'last_discriminator.pth')
            torch.save(self.autoenc, self.t_params['models_path'] + 'last_autoencoder.pth')
            torch.save(self.plotter, self.t_params['plot_path'] + 'last_plot.pth')
            torch.save(self, self.t_params['models_path'] + 'last_trainer_state.pth')

        logging.info('Training ended. Terminating gracefully..')

    @staticmethod
    def continue_training(last_trainer_state_path, with_params):

        last_trainer_state = os.path.join(last_trainer_state_path + 'last_trainer_state.pth')
        logging.info('Loading last trainer state in ' + last_trainer_state)
        trainer = torch.load(last_trainer_state)

        if with_params is not None:
            trainer.t_params = with_params
            trainer.use_cuda = with_params['use_cuda'] and torch.cuda.is_available()
            trainer.gpus_count = query_available_gpus() if trainer.use_cuda else 0
            if with_params['force-gpu-count'] > 0:
                trainer.gpus_count = with_params['force-gpu-count']
            if trainer.gpus_count == 0:
                trainer.use_cuda = False

            if trainer.t_params['training_samples_per_epoch'] == 0:
                trainer.t_params['training_samples_per_epoch'] = 2 ** 32
            if trainer.t_params['validation_samples_per_epoch'] == 0:
                trainer.t_params['validation_samples_per_epoch'] = 2 ** 32

            trainer.lambda_e_max = with_params['autoenc_loss_reg']
            trainer.lambda_e_step_size = (trainer.lambda_e_max - trainer.lambda_e) / with_params['autoenc_loss_reg_adaption_steps']
            trainer.gradient_max_norm = with_params['gradient_max_norm']

        params_state_title = 'original parameters.' if with_params is None else 'new trainer_params.'
        logging.info('Continuing experiment from epoch #%i with ' % (trainer.total_epochs+1) + params_state_title)
        logging.info('============Parameters:============')

        for key, value in trainer.t_params.items():
            logging.info(str(key) + ': ' + str(value))

        logging.info('===================================')

        trainer.train()

