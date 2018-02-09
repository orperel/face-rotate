import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import tarfile
import zipfile
import requests
from enum import Enum
from matplotlib import pyplot as plt


NN_INPUT_SIZE = 256              # Input size to the Fader NN
ENTRIES_PER_OUTPUT = 50000       # Amount of entries iterated for each batch
ANNOTATIONS_FILE = 'list_attr_celeba.txt'
IMAGES_FILE = 'img_align_celeba.zip'
data_root = '.'  # Where should the dataset be downloaded / extracted to


class DataPurpose(Enum):
    TRAINING = ('training', 0, 162770)            # 0..162770
    VALIDATION = ('validation', 162771, 182637)   # 162771..182637
    TEST = ('test', 182638, 202599)               # 182638..202599

def filename_from_url(url):
    begin = url.rfind('/') +1
    end = url.rfind('?')
    filename = url[begin:end]
    return filename

def isFileExists(url, local_path):
    filename = filename_from_url(url)
    return os.path.exists(os.path.join(local_path, filename))

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def fetch_remote_dataset(remote_filename):

    local_path = os.path.join(data_root, 'downloads', 'celebA')

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    elif isFileExists(remote_filename, local_path):
        print('%s already downloaded.' % (filename_from_url(remote_filename)))
        return  # Dataset already fetched
    print('Downloading dataset file from ' + remote_filename + '..')

    if remote_filename == IMAGES_FILE:
        filename = download_celeb_a(local_path)
    else:
        filename = download_celeb_a_txt(local_path, remote_filename)

    if str(filename).endswith('tar'):
        print('\nExtracting dataset to ' + local_path)
        tar = tarfile.open(filename)
        tar.extractall(path=local_path)
        tar.close()
        os.remove(filename)
    elif str(filename).endswith('zip') or str(filename).endswith('7z'):
        print('\nExtracting dataset to ' + local_path)
        zip = zipfile.ZipFile(filename, 'r')
        zip.extractall(path=local_path)
        zip.close()
        os.remove(filename)

    print('%s fetched successfully!' % (str(filename)))


def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return

    filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(save_path)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def download_celeb_a_txt(dirpath, filename):

    drive_id  = "0B7EVK8r0v71pblRyaVFSWGxPY0U"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    if save_path.endswith('zip'):
        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(save_path)
        os.rename(os.path.join(dirpath, zip_dir), dirpath)

def save_labels(labels_array, target_dir):
    tensor_labels = torch.from_numpy(labels_array)
    print("Saving celebA-labels to %s ..." % (os.path.join('.', target_dir)))

    labels_filename = 'celebA_labels.pth'
    labels_path = os.path.join(target_dir, labels_filename)
    torch.save(tensor_labels, labels_path)


def save_dataset(data_array, start_entry, target_dir):
    tensor_data = torch.from_numpy(data_array)
    print("Saving celebA-dataset to %s ..." % (os.path.join('.', target_dir)))

    start = start_entry
    end = start_entry + len(tensor_data)
    data_filename = 'celebA_%i_%i.pth' % (start, end)
    data_path = os.path.join(target_dir, data_filename)
    torch.save(tensor_data, data_path)

def extract_data(root_path, data_purpose):
    downloads_path = os.path.join(root_path, 'downloads', 'celebA')
    assert os.path.isdir(downloads_path), "Invalid downloads_path supplied for celebA dataset: %r" % downloads_path

    target_path = os.path.join(root_path, 'dataset', data_purpose.value[0])
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    target_path = os.path.join(target_path, 'celebA')
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    annotationsFile = 'list_attr_celeba.txt'
    imgs = []
    img_labels = []
    total_imgs = 0
    with open(os.path.join(downloads_path, annotationsFile), 'r', encoding='utf-8') as annotations:

        num_images = int(annotations.readline().strip())
        attributes = annotations.readline().strip().split(" ")
        print("{} attributes".format(len(attributes)))
        print("{}".format(attributes))
        img_files = np.loadtxt(annotations, usecols=[0], dtype=np.str)
        annotations.seek(0)
        male_label = np.loadtxt(annotations, usecols=[attributes.index("Male") + 1], dtype=np.int, skiprows=2) > 0

        for row_idx, row in enumerate(img_files[data_purpose.value[1]:data_purpose.value[2]]):

            filename = os.path.join(downloads_path, 'celebA', row)
            if not os.path.isfile(filename):
                print("Bad image file in dataset: %r" % filename)
                continue
            img = cv2.imread(filename)
            cropped = img[40:218, 0:178]
            scaled = cv2.resize(src=cropped, dsize=(NN_INPUT_SIZE, NN_INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
            imgs.append(scaled)

            label_idx = row_idx+data_purpose.value[1]
            male, female = int(male_label[label_idx]), int(not male_label[label_idx])
            img_labels.append(np.array([male, female]))

            if row_idx > 0 and row_idx % 2500 == 0:
                print('Iterated %i samples' % row_idx)
            if row_idx > 0 and row_idx % ENTRIES_PER_OUTPUT == 0:
                batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in imgs])

                save_dataset(data_array=batch, start_entry=total_imgs, target_dir=target_path)
                imgs = []
                total_imgs += len(batch)

        if imgs:
            batch = np.concatenate([img_data.transpose((2, 0, 1))[None] for img_data in imgs])
            save_dataset(data_array=batch, start_entry=total_imgs, target_dir=target_path)

        labels_batch = np.concatenate([label[None] for label in img_labels])
        save_labels(labels_array=labels_batch, target_dir=target_path)

    print('Extracted total of %i samples' % total_imgs)


fetch_remote_dataset(remote_filename=ANNOTATIONS_FILE)
fetch_remote_dataset(remote_filename=IMAGES_FILE)

extract_data(root_path=data_root, data_purpose=DataPurpose.TRAINING)
extract_data(root_path=data_root, data_purpose=DataPurpose.VALIDATION)
extract_data(root_path=data_root, data_purpose=DataPurpose.TEST)
