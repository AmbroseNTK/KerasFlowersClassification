
import glob
import os
import numpy as np
from PIL import Image
# Some of the flowers data is stored as .mat files
from scipy.io import loadmat
from shutil import copyfile
import sys
import tarfile
import time

# Loat the right urlretrieve based on python version
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import zipfile

# Useful for being able to dump images into the Notebook
import IPython.display as D

# Import CNTK and helpers
import cntk as C

data_root = os.path.join('..', 'Image')

datasets_path = os.path.join(data_root, 'DataSets')
output_path = os.path.join('.', 'temp', 'Output')


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_unless_exists(url, filename, max_retries=3):
    '''Download the file unless it already exists, with retry. Throws if all retries fail.'''
    if os.path.exists(filename):
        print('Reusing locally cached: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        retry_cnt = 0
        while True:
            try:
                urlretrieve(url, filename)
                print('Download completed.')
                return
            except:
                retry_cnt += 1
                if retry_cnt == max_retries:
                    print('Exceeded maximum retry count, aborting.')
                    raise
                print('Failed to download, retrying.')
                time.sleep(np.random.randint(1, 10))


def write_to_file(file_path, img_paths, img_labels):
    with open(file_path, 'w+') as f:
        for i in range(0, len(img_paths)):
            f.write('%s\t%s\n' %
                    (os.path.abspath(img_paths[i]), img_labels[i]))


def download_flowers_dataset(dataset_root=os.path.join(datasets_path, 'Flowers')):
    ensure_exists(dataset_root)
    flowers_uris = [
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
    ]
    flowers_files = [
        os.path.join(dataset_root, '102flowers.tgz'),
        os.path.join(dataset_root, 'imagelabels.mat'),
        os.path.join(dataset_root, 'setid.mat')
    ]
    for uri, file in zip(flowers_uris, flowers_files):
        download_unless_exists(uri, file)
    tar_dir = os.path.join(dataset_root, 'extracted')
    if not os.path.exists(tar_dir):
        print('Extracting {} to {}'.format(flowers_files[0], tar_dir))
        os.makedirs(tar_dir)
        tarfile.open(flowers_files[0]).extractall(path=tar_dir)
    else:
        print('{} already extracted to {}, using existing version'.format(
            flowers_files[0], tar_dir))

    flowers_data = {
        'data_folder': dataset_root,
        'training_map': os.path.join(dataset_root, '6k_img_map.txt'),
        'testing_map': os.path.join(dataset_root, '1k_img_map.txt'),
        'validation_map': os.path.join(dataset_root, 'val_map.txt')
    }

    if not os.path.exists(flowers_data['training_map']):
        print('Writing map files ...')
        # get image paths and 0-based image labels
        image_paths = np.array(
            sorted(glob.glob(os.path.join(tar_dir, 'jpg', '*.jpg'))))
        image_labels = loadmat(flowers_files[1])['labels'][0]
        image_labels -= 1

        # read set information from .mat file
        setid = loadmat(flowers_files[2])
        idx_train = setid['trnid'][0] - 1
        idx_test = setid['tstid'][0] - 1
        idx_val = setid['valid'][0] - 1

        # Confusingly the training set contains 1k images and the test set contains 6k images
        # We swap them, because we want to train on more data
        write_to_file(flowers_data['training_map'],
                      image_paths[idx_train], image_labels[idx_train])
        write_to_file(flowers_data['testing_map'],
                      image_paths[idx_test], image_labels[idx_test])
        write_to_file(flowers_data['validation_map'],
                      image_paths[idx_val], image_labels[idx_val])
        print('Map files written, dataset download and unpack completed.')
    else:
        print('Using cached map files.')

    return flowers_data


download_flowers_dataset()
