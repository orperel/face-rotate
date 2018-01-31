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
    TRAINING = 'training',
    VALIDATION = 'validation',
    TEST = 'test'


def fetch_remote_dataset(remote_url):
    '''
    UMDFaces dataset:
    https://arxiv.org/pdf/1611.01484v2.pdf
    '''

    if 'batch1' in remote_url:
        index = '1'
    elif 'batch2' in remote_url:
        index = '2'
    elif 'batch3' in remote_url:
        index = '3'
    else:
        raise ValueError('Invalid batch index for UMD batch')

    local_path = os.path.join(data_root, 'downloads', 'umdfaces_batch' + index)

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    elif os.stat(local_path).st_size > 0:
        print('Batch %s already downloaded.' % (index))
        return  # Dataset already fetched
    print('Downloading dataset from ' + remote_url + '..')
    filename = wget.download(url=remote_url)

    print('\nExtracting dataset to ' + local_path)
    tar = tarfile.open(filename)
    tar.extractall(path=local_path)
    tar.close()
    os.remove(filename)
    print('Dataset ready!')


def save_labels(labels_array, data_name, batchIndex, target_dir):
    tensor_labels = torch.from_numpy(labels_array)
    print("Saving %s-labels to %s ..." % (data_name, os.path.join('.', target_dir)))

    labels_filename = 'umd_%i_%s_labels.pth' % (batchIndex, data_name)
    labels_path = os.path.join(target_dir, labels_filename)
    torch.save(tensor_labels, labels_path)


def save_dataset(data_array, data_name, batchIndex, start_entry, target_dir):
    tensor_data = torch.from_numpy(data_array)
    print("Saving %s-dataset to %s ..." % (data_name, os.path.join('.', target_dir)))

    start = start_entry
    end = start_entry + len(tensor_data)
    data_filename = 'umd_%i_%s_%i_%i.pth' % (batchIndex, data_name, start, end)
    data_path = os.path.join(target_dir, data_filename)
    torch.save(tensor_data, data_path)


def normalize_dof(yaw, pitch, roll):
    yaw_n = float(yaw + 90) / 180
    pitch_n = float(pitch + 90) / 180
    roll_n = float(roll + 90) / 180

    return yaw_n, pitch_n, roll_n

def extract_data(root_path, batch_index, data_purpose):
    downloads_path = os.path.join(root_path, 'downloads', 'umdfaces_batch')
    downloads_path += str(batch_index)
    downloads_path = os.path.join(downloads_path, 'umdfaces_batch' + str(batch_index))
    assert os.path.isdir(downloads_path), "Invalid downloads_path supplied for UMDFaces dataset: %r" % downloads_path

    target_path = os.path.join(root_path, 'dataset', data_purpose.value[0])
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
            img = cv2.imread(filename)
            x, y = round(float(row['FACE_X'])), round(float(row['FACE_Y']))
            width, height = round(float(row['FACE_WIDTH'])), round(float(row['FACE_HEIGHT']))
            max_dim = max(width, height)
            cropped = img[y:y+max_dim, x:x+max_dim]

            yaw, pitch, roll = round(float(row['YAW'])), round(float(row['PITCH'])), round(float(row['ROLL']))
            yaw, pitch, roll = normalize_dof(yaw, pitch, roll)

            if not (1.0 >= yaw >= 0.0 and 1.0 >= pitch >= 0.0 and 1.0 >= roll >= 0.0):
                print('Yaw-Pitch-Roll values for sample exceed the normal range (%f, %f, %f) - skipping' % (yaw, pitch, roll))
            else:
                if max_dim < NN_INPUT_SIZE:
                    scaled = cv2.resize(src=cropped, dsize=(NN_INPUT_SIZE, NN_INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
                    enlarged.append(scaled)
                    enlarged_labels.append(np.array([yaw, pitch, roll]))
                elif max_dim > NN_INPUT_SIZE:
                    scaled = cv2.resize(src=cropped, dsize=(NN_INPUT_SIZE, NN_INPUT_SIZE), interpolation=cv2.INTER_AREA)
                    decimated.append(scaled)
                    decimated_labels.append(np.array([yaw, pitch, roll]))

            if row_idx > 0 and row_idx % 2500 == 0:
                print('Iterated %i samples [%i decimated, %i enlarged]' % (row_idx, len(decimated), len(enlarged)))
            if row_idx > 0 and row_idx % ENTRIES_PER_OUTPUT == 0:
                decimated_batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in decimated])
                enlarged_batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in enlarged])
                all_batch = np.concatenate((decimated_batch, enlarged_batch), axis=0)

                save_dataset(data_array=decimated_batch, start_entry=total_decimated, data_name='decimated',
                             batchIndex=batch_index, target_dir=target_path_decimated)
                save_dataset(data_array=enlarged_batch, start_entry=total_enlarged, data_name='enlarged',
                             batchIndex=batch_index, target_dir=target_path_enlarged)
                save_dataset(data_array=all_batch, start_entry=total_decimated+total_enlarged, data_name='all',
                             batchIndex=batch_index, target_dir=target_path_all)

                decimated = []
                enlarged = []
                total_decimated += len(decimated_batch)
                total_enlarged += len(enlarged_batch)

        # Last batch
        if decimated:
            decimated_batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in decimated])
            save_dataset(data_array=decimated_batch, start_entry=total_decimated, data_name='decimated',
                         batchIndex=batch_index, target_dir=target_path_decimated)
        if enlarged:
            enlarged_batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in enlarged])
            save_dataset(data_array=enlarged_batch, start_entry=total_enlarged, data_name='enlarged',
                         batchIndex=batch_index, target_dir=target_path_enlarged)
        if decimated or enlarged:
            all_batch = np.concatenate((decimated_batch, enlarged_batch), axis=0)
            save_dataset(data_array=all_batch, start_entry=total_decimated + total_enlarged, data_name='all',
                         batchIndex=batch_index, target_dir=target_path_all)

        decimated_labels_batch = np.concatenate([label[None] for label in decimated_labels])
        enlarged_labels_batch = np.concatenate([label[None] for label in enlarged_labels])
        all_labels_batch = np.concatenate((decimated_labels_batch, enlarged_labels_batch), axis=0)

        total_decimated += len(decimated_batch)
        total_enlarged += len(enlarged_batch)

        save_labels(labels_array=decimated_labels_batch, data_name='decimated',
                    batchIndex=batch_index, target_dir=target_path_decimated)
        save_labels(labels_array=enlarged_labels_batch, data_name='enlarged',
                    batchIndex=batch_index, target_dir=target_path_enlarged)
        save_labels(labels_array=all_labels_batch, data_name='all',
                    batchIndex=batch_index, target_dir=target_path_all)

    print('Extracted %i enlarged and %i decimated samples. Total of %i' %
          (total_enlarged, total_decimated, total_decimated+total_enlarged))


fetch_remote_dataset(remote_url=UMD_BATCH1_URL)
fetch_remote_dataset(remote_url=UMD_BATCH2_URL)
fetch_remote_dataset(remote_url=UMD_BATCH3_URL)
extract_data(root_path=data_root, batch_index=1, data_purpose=DataPurpose.TRAINING)
extract_data(root_path=data_root, batch_index=2, data_purpose=DataPurpose.VALIDATION)
extract_data(root_path=data_root, batch_index=3, data_purpose=DataPurpose.TEST)
