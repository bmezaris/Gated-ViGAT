import os
import sys
import numpy as np
import csv
from torch.utils.data import Dataset


class FCVID(Dataset):
    NUM_CLASS = 239
    NUM_FRAMES = 9
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'VIT':
            self.local_folder = 'vit_local'
            self.global_folder = 'vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'RESNET':
            self.local_folder = 'R152_local'
            self.global_folder = 'R152_global'
            self.NUM_FEATS = 2048
        else:
            sys.exit("Unknown Extractor")

        split_path = os.path.join(root_dir, 'materials', 'FCVID_VideoName_TrainTestSplit.txt')
        data_split = np.genfromtxt(split_path, dtype='str')

        label_path = os.path.join(root_dir, 'materials', 'FCVID_Label.txt')
        labels = np.genfromtxt(label_path, dtype=np.float32)

        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[1] == self.phase:
                base, _ = os.path.splitext(os.path.normpath(row[0]))
                feats_path = os.path.join(root_dir, self.local_folder, base + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.labels = labels[mask, :]
        self.videos = data_split[mask, 0]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        return feats, feat_global, label, name


class ACTNET(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 120
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'VIT':
            self.local_folder = 'feats/vit_local'
            self.global_folder = 'feats/vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'RESNET':
            self.local_folder = 'R152_local'
            self.global_folder = 'R152_global'
            self.NUM_FEATS = 2048
        else:
            sys.exit("Unknown Extractor")

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'actnet_train_split.txt')
        else:
            split_path = os.path.join(root_dir, 'actnet_val_split.txt')

        vidname_list = []
        labels_list = []
        with open(split_path) as f:
            for line in f:
                row = line.strip().split(',')
                vidname_list.append(row[0])
                labels_list.append(list(map(int, row[2:])))

        length = len(vidname_list)
        labels_np = np.zeros((length, self.NUM_CLASS), dtype=np.float32)
        for i, lbllst in enumerate(labels_list):
            for lbl in lbllst:
                labels_np[i, lbl] = 1.

        self.labels = labels_np
        self.videos = vidname_list

        self.num_missing = 0  # no missing videos by default!!

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')  #
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')  #
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        return feats, feat_global, label, name


class miniKINETICS(Dataset):
    NUM_CLASS = 200
    NUM_FRAMES = 120
    NUM_BOXES = 50

    def __init__(self, root_dir, is_train, ext_method):
        self.root_dir = root_dir
        self.phase = 'train' if is_train else 'test'
        if ext_method == 'VIT':
            self.local_folder = 'feats/vit_local'
            self.global_folder = 'feats/vit_global'
            self.NUM_FEATS = 768
        elif ext_method == 'RESNET':
            self.local_folder = 'R152_local'
            self.global_folder = 'R152_global'
            self.NUM_FEATS = 2048
        else:
            sys.exit("Unknown Extractor")

        if self.phase == 'train':
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130trainv2.csv')
        else:
            split_path = os.path.join(root_dir, 'annotations', 'miniKinetics130val.csv')

        vidname_list = []
        labels_list = []
        self.num_missing = 0

        with open(split_path) as f:
            file = csv.reader(f)
            header = []
            header = next(file)
            if self.phase == 'train':
                mask = np.zeros(121215, dtype=bool)  #80000
            else:
                mask = np.zeros(9867, dtype=bool)    #5000
            for i, row in enumerate(file):
                base = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6) + '_frames'
                vidname_list.append(base)
                labels_list.append(list(map(int, [row[0]])))
                feats_path = os.path.join(root_dir, self.local_folder, base + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1
        # length = len(vidname_list)
        # labels_np = np.zeros((length, self.NUM_CLASS), dtype=np.float32)
        # for i, lbllst in enumerate(labels_list):
        #     for lbl in lbllst:
        #         labels_np[i, lbl] = 1.
        #
        # self.labels = labels_np[mask]
        self.labels = np.array(labels_list, dtype=np.int64).squeeze()[mask]   # , :]
        self.videos = np.array(vidname_list)[mask]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        # name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, self.local_folder, name + '.npy')  #
        global_path = os.path.join(self.root_dir, self.global_folder, name + '.npy')  #
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx]

        return feats, feat_global, label, name
