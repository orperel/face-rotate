import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import time
import logging
import threading


class SubGroupsRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.prefetch_thread = None
        self.last_file_visited = None

    def __iter__(self):
        file_order = np.random.permutation(self.data_source.data_files)

        if len(file_order) > 1 and self.last_file_visited is not None:  # Start from currently cached batch
            last_visited_idx = np.where(file_order==self.last_file_visited)
            file_order[0], file_order[last_visited_idx] = file_order[last_visited_idx], file_order[0]

        for file_idx, group_file in enumerate(file_order):
            group_size = int(group_file[1]) - int(group_file[0])
            self.last_file_visited = group_file
            for count, idx in enumerate(np.random.permutation(group_size)):
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
        self.dataset_lock = threading.Lock()
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
        self.prev_batch_range = (-1, -1)
        self.prev_batch = None

    @staticmethod
    def normalize_img(x):
        return x.float().div(255).mul_(2).sub_(1)  # To range -1 to 1

    @staticmethod
    def flip_horizontally(x, y):
        x = x.index_select(2, torch.arange(x.size(2) - 1, -1, -1).long())
        return x,y

    def prefetch(self, idx, true_prefetch):
        with self.dataset_lock:
            if not self.current_batch_range[0] <= idx < self.current_batch_range[1]:
                if self.prev_batch_range[0] <= idx < self.prev_batch_range[1]:
                    return self.fetch_from_batch(self.prev_batch, self.prev_batch_range, idx)
                next_data_entry = next(entry for entry in self.data_files if entry[0] <= idx < entry[1])

                self.prev_batch = self.current_batch
                self.prev_batch_range = self.current_batch_range
                self.current_batch_range = (next_data_entry[0], next_data_entry[1])

                if true_prefetch:
                    logging.info('Prefetch optimization executed..')
                else:
                    logging.info('Hot cache miss for data-batch file..')
                logging.info('Swapping data-batch file to: ' + next_data_entry[2])
                start = time.time()

                self.current_batch = torch.load(next_data_entry[2])
                end = time.time()
                logging.info('Loading completed after ' + '{0:.2f}'.format(end - start) + ' seconds')

    def fetch_from_batch(self, batch_file, batch_range, idx):
        x = batch_file[idx - batch_range[0]]
        x = self.normalize_img(x)
        y = self.labels[idx].float()

        if self.h_flip_augment and random.random() >= 0.5:
            x, y = self.flip_horizontally(x, y)

        return {'data': x, 'label': y}

    def __getitem__(self, idx):
        # Load next batch if needed,
        # Maintain a LRU history of size 1
        if not self.current_batch_range[0] <= idx < self.current_batch_range[1]:
            if self.prev_batch_range[0] <= idx < self.prev_batch_range[1]:
                return self.fetch_from_batch(self.prev_batch, self.prev_batch_range, idx)
            else:
                self.prefetch(idx, False)

        return self.fetch_from_batch(self.current_batch, self.current_batch_range, idx)

    def __len__(self):
        return self.data_files[-1][1]