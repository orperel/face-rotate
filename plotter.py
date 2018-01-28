import os, errno
import visdom


# Visdom
# https://github.com/facebookresearch/visdom
#
# For lose report use:
# python -m visdom.server
# http://localhost:8097
class Plotter:
    def __init__(self, path):
        '''
        :param path: path for plots
        '''
        self.path = path
        self.vis = visdom.Visdom()
        self.colors = ['blue', 'red', 'green', 'yellow', 'orange', 'pink', 'brown']
        self.last_color_ptr = 0

        if path is not None:
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.loss_plot_data = {}

    def update_loss_plot_data(self, mode, new_epoch, new_loss):

        if mode not in self.loss_plot_data:
            color = self.colors[self.last_color_ptr]
            self.last_color_ptr += 1
            self.loss_plot_data[mode] = {'losses': [], 'epochs': [], 'name': mode, 'color': color}

        self.loss_plot_data[mode]['epochs'].append(new_epoch)
        self.loss_plot_data[mode]['losses'].append(new_loss)

    def plot_losses(self, window):
        traces = []
        for name, data in self.loss_plot_data.items():
            new_trace = dict(x=data['epochs'], y=data['losses'], mode="lines", type='custom',
                             marker={'color': data['color']}, name=data['name'])
            traces.append(new_trace)
        layout = dict(title="Loss", xaxis={'title': 'Epoch [#]'}, yaxis={'title': 'Loss'})
        self.vis._send({'data': traces, 'layout': layout, 'win': window})