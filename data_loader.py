import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class UMDDataset(Dataset):

    def __init__(self, path, ypr_quant, deg_dim, use_cuda):

        assert ypr_quant and (180 % deg_dim == 0), "Invalid deg_dim parameter defined for trainer params: %r" % deg_dim

        self.ypr_quant = ypr_quant
        self.deg_dim = deg_dim
        self.deg_dim_quant = 180 / deg_dim  # How much quantization should be applied
        self.use_cuda = use_cuda
        self.data_files = []
        self.labels = None
        batches = []

        for filename in os.listdir(path):
            assert filename.startswith('umd_') and filename.endswith('.pth'), \
                   "Invalid file %r, does not belong to UMDFaces dataset" % filename

            filepath = os.path.join(path, filename)

            if 'labels' in filename:
                self.labels = torch.load(filepath)
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
        return x.float().div_(255).mul_(2).sub_(1) # To range -1 to 1

    @staticmethod
    def normalize_angles(y):
        return y.mul_(180)

    @staticmethod
    def to_one_hot(y, dof=180, dof_quant=1):
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.div_(dof_quant).type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(y_tensor.size()[0], dof).scatter_(1, y_tensor, 1)
        return y_one_hot.view(-1)

    def __getitem__(self, idx):
        # Load next batch if needed
        if not self.current_batch_range[0] <= idx < self.current_batch_range[1]:
            next_data_entry = next(entry for entry in self.data_files if entry[0] <= idx < entry[1])
            self.current_batch_range = (next_data_entry[0], next_data_entry[1])
            self.current_batch = torch.load(next_data_entry[2])

        x = self.current_batch[idx - self.current_batch_range[0]]
        x = self.normalize_img(x)
        y = self.labels[idx]
        y = self.normalize_angles(y)
        y = self.to_one_hot(y, self.deg_dim, self.deg_dim_quant)

        if self.use_cuda:
            x.cuda()
            y.cuda()

        return {'data': x, 'label': y}

    def __len__(self):
        return self.data_files[-1][1]