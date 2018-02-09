import os
import csv
import numpy as np
import torch
import cv2
import wget
import tarfile
from enum import Enum


NN_INPUT_SIZE = 256              # Input size to the Fader NN
ENTRIES_PER_OUTPUT = 40000       # Amount of entries iterated for each batch
UMD_BATCH1_URL = 'https://obj.umiacs.umd.edu/umdfaces/umdfaces_images/umdfaces_batch1.tar.gz'
UMD_BATCH2_URL = 'https://obj.umiacs.umd.edu/umdfaces/umdfaces_images/umdfaces_batch2.tar.gz'
UMD_BATCH3_URL = 'https://obj.umiacs.umd.edu/umdfaces/umdfaces_images/umdfaces_batch3.tar.gz'
data_root = '/mnt/data/orperel'  # Where should the dataset be downloaded / extracted to


class DataPurpose(Enum):
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'


def save_labels(labels_array, data_name, batchIndex, target_dir):
    tensor_labels = torch.from_numpy(labels_array).float()
    print("Saving %s-labels to %s ..." % (data_name, os.path.join('.', target_dir)))

    labels_filename = 'umd_%i_%s_gender_labels.pth' % (batchIndex, data_name)
    labels_path = os.path.join(target_dir, labels_filename)
    torch.save(tensor_labels, labels_path)


def extract_data(root_path, batch_index, data_purpose):
    downloads_path = os.path.join(root_path, 'downloads', 'umdfaces_batch')
    downloads_path += str(batch_index)
    downloads_path = os.path.join(downloads_path, 'umdfaces_batch' + str(batch_index))
    assert os.path.isdir(downloads_path), "Invalid downloads_path supplied for UMDFaces dataset: %r" % downloads_path

    target_path = os.path.join(root_path, 'dataset', data_purpose.value)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    target_path_all = os.path.join(target_path, 'all')
    target_path_decimated = os.path.join(target_path, 'decimated')
    target_path_enlarged = os.path.join(target_path, 'enlarged')
    if not os.path.isdir(target_path_all):
        os.makedirs(target_path_all)
    if not os.path.isdir(target_path_decimated):
        os.makedirs(target_path_decimated)
    if not os.path.isdir(target_path_enlarged):
        os.makedirs(target_path_enlarged)

    annotationsFile = 'umdfaces_batch%i_ultraface.csv' % batch_index
    decimated = []
    decimated_labels = []
    enlarged = []
    enlarged_labels = []
    total_decimated = 0
    total_enlarged = 0
    with open(os.path.join(downloads_path, annotationsFile), 'r', encoding='utf-8') as annotationsCsv:

        reader = csv.DictReader(annotationsCsv)
        for row_idx, row in enumerate(reader):

            filename = os.path.join(downloads_path, row['FILE'])
            if not os.path.isfile(filename):
                print("Bad image file in dataset: %r" % filename)
                continue


            male, female = round(float(row['PR_MALE'])), round(float(row['PR_FEMALE']))
            enlarged_labels.append(np.array([male, female]))
            decimated_labels.append(np.array([male, female]))

            if row_idx > 0 and row_idx % 2500 == 0:
                print('Iterated %i samples [%i decimated, %i enlarged]' % (row_idx, len(decimated), len(enlarged)))

        decimated_labels_batch = np.concatenate([label[None] for label in decimated_labels])
        enlarged_labels_batch = np.concatenate([label[None] for label in enlarged_labels])
        all_labels_batch = np.concatenate((decimated_labels_batch, enlarged_labels_batch), axis=0)

        save_labels(labels_array=all_labels_batch, data_name='all',
                    batchIndex=batch_index, target_dir=target_path_all)

    print('Extracted %i enlarged and %i decimated samples. Total of %i' %
          (total_enlarged, total_decimated, total_decimated+total_enlarged))


extract_data(root_path=data_root, batch_index=1, data_purpose=DataPurpose.TRAINING)
extract_data(root_path=data_root, batch_index=2, data_purpose=DataPurpose.VALIDATION)
extract_data(root_path=data_root, batch_index=3, data_purpose=DataPurpose.TEST)
