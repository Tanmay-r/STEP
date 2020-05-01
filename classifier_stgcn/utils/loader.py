# sys
import h5py
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils import common
from utils.mocap_dataset import MocapDataset
# torch
import torch
from torchvision import datasets, transforms
from imblearn.over_sampling import SMOTE
from collections import Counter
import operator

# Load data from the MPI dataset
def load_data_MPI(_path, _ftype, coords, joints, cycles=3):

    # Counts: 'Hindi': 292, 'German': 955, 'English': 200
    bvhDirectory = os.path.join(_path, "bvh")
    tagDirectory = os.path.join(_path, "tags")

    data_list = {}
    num_samples = 0
    time_steps = 0
    labels_list = {}
    fileIDs = []
    for filenum in range(1, 1452):
        filename = str(filenum).zfill(6)
        # print(filenum)
        if not os.path.exists(os.path.join(tagDirectory, filename + ".txt")):
            print(os.path.join(tagDirectory, filename + ".txt"), " not found!")
            continue
        names, parents, offsets, positions, rotations = MocapDataset.load_bvh(os.path.join(bvhDirectory, filename + ".bvh"))
        tag, text = MocapDataset.read_tags(os.path.join(tagDirectory, filename + ".txt"))
        num_samples += 1
        positions = np.reshape(positions, (positions.shape[0], positions.shape[1]*positions.shape[2]))
        data_list[filenum] = list(positions)
        time_steps_curr = len(positions)
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        if "Hindi" in tag:
            labels_list[filenum] = 0
        elif "German" in tag:
            labels_list[filenum] = 1
        elif "English" in tag:
            labels_list[filenum] = 2
        else:
            print("ERROR: ", tag)
        fileIDs.append(filenum)

    
    labels = np.empty(num_samples)
    data = np.empty((num_samples, time_steps*cycles, joints*coords))
    index = 0
    for si in fileIDs:
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
        if (data_list_curr.shape[1] != 69):
            continue
        for ci in range(cycles):
            data[index, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
        labels[index] = labels_list[si]
        index += 1
    data = data[:index]
    labels = labels[:index]
    print(index, num_samples)
        
    # data = common.get_affective_features(np.reshape(data, (data.shape[0], data.shape[1], joints, coords)))[:, :, :48]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1)
    data_train, labels_train = balance_classes(data_train, labels_train)   

    return data, labels, data_train, labels_train, data_test, labels_test


def balance_classes(data_train, labels_train):
    print("Initial distribution: ", Counter(labels_train))
    orig_shape = data_train.shape
    print("Shape of data_train:", orig_shape)

    data_train = np.reshape(data_train, (data_train.shape[0], data_train.shape[1]*data_train.shape[2]))
    print("Shape of data_train (2D):", data_train.shape)

    sampling_strategy = Counter(labels_train)
    maxID = max(sampling_strategy.items(), key=operator.itemgetter(1))[0]
    for key in sampling_strategy:
        sampling_strategy[key] = sampling_strategy[maxID]
    print("Expected distribution: ", sampling_strategy)

    sm = SMOTE(sampling_strategy=sampling_strategy)
    data_train, labels_train = sm.fit_sample(data_train, labels_train)
    print("Final distribution: ", Counter(labels_train))

    new_shape = (Counter(labels_train)[labels_train[0]]*3, orig_shape[1], orig_shape[2])
    data_train = np.reshape(data_train, new_shape)
    print("Shape of data_train (3D):", data_train.shape)
    return data_train, labels_train

def load_data(_path, _ftype, coords, joints, cycles=3):

    file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
    ff = h5py.File(file_feature, 'r')
    file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
    fl = h5py.File(file_label, 'r')

    data_list = []
    num_samples = len(ff.keys())
    time_steps = 0
    labels = np.empty(num_samples)
    for si in range(num_samples):
        ff_group_key = list(ff.keys())[si]
        data_list.append(list(ff[ff_group_key]))  # Get the data
        time_steps_curr = len(ff[ff_group_key])
        if time_steps_curr > time_steps:
            time_steps = time_steps_curr
        labels[si] = fl[list(fl.keys())[si]][()]

    print(len(data_list[0]))
    print(len(data_list[0][0]))
    data = np.empty((num_samples, time_steps*cycles, joints*coords))
    for si in range(num_samples):
        data_list_curr = np.tile(data_list[si], (int(np.ceil(time_steps / len(data_list[si]))), 1))
        for ci in range(cycles):
            data[si, time_steps * ci:time_steps * (ci + 1), :] = data_list_curr[0:time_steps]
    data = common.get_affective_features(np.reshape(data, (data.shape[0], data.shape[1], joints, coords)))[:, :, :48]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1)
    return data, labels, data_train, labels_train, data_test, labels_test


def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

    def __init__(self, data, label, joints, coords, num_classes):
        # data: N C T J
        self.data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords, 1))
        self.data = np.moveaxis(self.data, [1, 2, 3], [2, 3, 1])

        # load label
        self.label = label

        self.N, self.C, self.T, self.J, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label
