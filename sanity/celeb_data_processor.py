import os
import csv
import numpy as np
import torch
import wget
from tqdm import tqdm
import tarfile
import zipfile
import requests
from enum import Enum


NN_INPUT_SIZE = 256              # Input size to the Fader NN
ENTRIES_PER_OUTPUT = 50000       # Amount of entries iterated for each batch
DATA_PARTITION_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxLE5t6HqyD8sQCmzWJRcHa/Eval/list_eval_partition.txt?dl=0'
ANNOTATIONS_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=0'
IMAGES_URL = 'https://dl.dropboxusercontent.com/content_link/QK78i6jtgHmTtPGRpPPZlpItFKdG7ucOhVf2VJlyIeNV6QrIp2wW4r0HU6aslA95/file?_download_id=50756820272987894656519570470714126039199463249934445303454475158&_notify_domain=www.dropbox.com&dl=1'
data_root = '.'  # Where should the dataset be downloaded / extracted to


class DataPurpose(Enum):
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

def filename_from_url(url):
    begin = url.rfind('/') +1
    end = url.rfind('?')
    filename = url[begin:end]
    return filename

def isFileExists(url, local_path):
    filename = filename_from_url(url)
    return os.path.exists(os.path.join(local_path, filename))

def fetch_remote_dataset(remote_url):

    local_path = os.path.join(data_root, 'downloads', 'celebA')

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    elif isFileExists(remote_url, local_path):
        print('%s already downloaded.' % (filename_from_url(remote_url)))
        return  # Dataset already fetched
    print('Downloading dataset file from ' + remote_url + '..')

    if remote_url == IMAGES_URL:
        filename = download_celeb_a(local_path)
    else:
        filename = wget.download(url=remote_url, out=local_path)

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


fetch_remote_dataset(remote_url=DATA_PARTITION_URL)
fetch_remote_dataset(remote_url=ANNOTATIONS_URL)
fetch_remote_dataset(remote_url=IMAGES_URL)