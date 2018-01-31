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

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.loss_plot_data, self.last_color_ptr, self.path)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.loss_plot_data, self.last_color_ptr, self.path = state
        self.vis = visdom.Visdom()
        self.colors = ['blue', 'red', 'green', 'yellow', 'orange', 'pink', 'brown']

    def update_loss_plot_data(self, network, mode, new_epoch, new_loss):
        if network not in self.loss_plot_data:
            self.loss_plot_data[network] = {}
        if mode not in self.loss_plot_data[network]:
            color = self.colors[self.last_color_ptr]
            self.last_color_ptr += 1
            self.loss_plot_data[network][mode] = {'losses': [], 'epochs': [], 'name': network + ' ' + mode, 'color': color}

        self.loss_plot_data[network][mode]['epochs'].append(new_epoch)
        self.loss_plot_data[network][mode]['losses'].append(new_loss)

    def plot_losses(self, window):
        traces = []
        network_traces = {}
        for network in self.loss_plot_data.keys():
            network_traces[network] = []
            for mode, data in self.loss_plot_data[network].items():
                new_trace = dict(x=data['epochs'], y=data['losses'], mode="lines", type='custom',
                                 marker={'color': data['color']}, name=data['name'])
                traces.append(new_trace)
                network_traces[network].append(new_trace)

        layout = dict(title="Loss", xaxis={'title': 'Epoch [#]'}, yaxis={'title': 'Loss'})
        self.vis._send({'data': traces, 'layout': layout, 'win': window})
        for network in network_traces:
            layout = dict(title=network + " Loss", xaxis={'title': 'Epoch [#]'}, yaxis={'title': 'Loss'})
            self.vis._send({'data': network_traces[network], 'layout': layout, 'win': network + ' Losses'})