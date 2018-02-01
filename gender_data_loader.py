import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import time
import logging


class SubGroupsRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        for group_file in np.random.permutation(self.data_source.data_files):
            group_size = int(group_file[1]) - int(group_file[0])
            for idx in np.random.permutation(group_size):
                yield int(group_file[0]) + idx

    def __len__(self):
        return len(self.data_source)


class UMDDataset(Dataset):

    def __init__(self, path, ypr_quant, ypr_regress, deg_dim, h_flip_augment, use_cuda):

        assert not ypr_quant or (180 % deg_dim == 0), \
            "Invalid deg_dim parameter defined for trainer params: %r" % deg_dim

        self.ypr_quant = ypr_quant
        self.ypr_regress = ypr_regress
        self.deg_dim = 2
        self.h_flip_augment = h_flip_augment
        self.use_cuda = use_cuda
        self.data_files = []
        self.labels = None
        batches = []

        for filename in os.listdir(path):
            assert filename.startswith('umd_') and filename.endswith('.pth'), \
                   "Invalid file %r, does not belong to UMDFaces dataset" % filename

            filepath = os.path.join(path, filename)

            if 'gender_labels' in filename:
                self.labels = torch.load(filepath)
            elif '_labels' in filename:
                pass    # Ignore
            else:
                last_underscore_idx = filename.rfind('_')
                batch_end = int(filename[last_underscore_idx+1:-len('.pth')])
                second_last_underscore_idx = filename[:last_underscore_idx].rfind('_')
                batch_start = int(filename[second_last_underscore_idx+1:last_underscore_idx])
                self.data_files.append((batch_start, batch_end, filepath))

        # Verify all files are present
        assert self.data_files, "No data files found for UMDFaces dataset in path %r" % path
        assert self.labels is not None, "No labels file found for UMDFaces dataset in path %r" % path

        self.data_files.sort(key=lambda tup: tup[0])
        assert self.data_files[0][0] == 0, "First data file for UMDFaces dataset is missing in path %r" % path
        for i, _ in enumerate(batches[:-1]):
            assert self.data_files[i][1] == self.data_files[i+1][0],\
                "Missing data file for batch starting from %r in path %r" % (batches[i+1][0], path)

        self.current_batch_range = (-1, -1)
        self.current_batch = None

    @staticmethod
    def normalize_img(x):
        return x.float().div(255).mul_(2).sub_(1)  # To range -1 to 1

    @staticmethod
    def flip_horizontally(x, y):
        x = x.index_select(2, torch.arange(x.size(2) - 1, -1, -1).long())
        return x,y

    def __getitem__(self, idx):
        # Load next batch if needed
        if not self.current_batch_range[0] <= idx < self.current_batch_range[1]:
            next_data_entry = next(entry for entry in self.data_files if entry[0] <= idx < entry[1])
            self.current_batch_range = (next_data_entry[0], next_data_entry[1])

            logging.info('Swapping data-batch file to: ' + next_data_entry[2])
            start = time.time()
            self.current_batch = torch.load(next_data_entry[2])
            end = time.time()
            logging.info('Loading completed after ' + '{0:.2f}'.format(end - start) + ' seconds')

        x = self.current_batch[idx - self.current_batch_range[0]]
        x = self.normalize_img(x)
        y = self.labels[idx]

        if self.h_flip_augment and random.random() >= 0.5:
            x, y = self.flip_horizontally(x, y)

        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()

        return {'data': x, 'label': y}

    def __len__(self):
        return self.data_files[-1][1]