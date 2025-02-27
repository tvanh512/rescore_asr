import torch
import glob, os
from natsort import natsorted
from datasets import DatasetDict , Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader
import json
import pandas as pd

def load_myst_original(data_dir, split ='development', num_samples = None):
    # Note some wav files doesn't have transcript, so we need to do filter
    labels = []
    labels_file = glob.glob(os.path.join(data_dir, 'data',split,'*','*','*.trn'))
    labels_file  = natsorted(labels_file)
    if num_samples:
        labels_file = labels_file[:num_samples]
    wavs_file = [label_file[:-3] + 'wav' for label_file in labels_file]
    for i in range(len(labels_file)):
        with open(labels_file[i], "r") as f:
            label = f.read()
            labels.append(label.replace("’","'"))
    return labels, wavs_file, labels_file

def load_myst_json(data_dir, json_path,split ='valid', num_samples = None):
    with open(os.path.join(json_path,split +'.json'), 'r') as json_file:
        data = json.load(json_file)
    data = list(data.values())
    wavs_file =[]
    labels_file = []
    labels = []
    labels_dir = os.path.join(json_path, 'transcript', split)
    for i in range(len(data)):
        wavs_file.append(data[i]['wav'].replace('../corpora/myst-v0.4.2',data_dir))
        labels.append(data[i]['transcript'])
        label_file = os.path.join(labels_dir,os.path.basename(wavs_file[i]).replace('.wav', '.txt'))
        labels_file.append(label_file)
    if num_samples:
        labels = labels[:num_samples]
        wavs_file = wavs_file[:num_samples]
        labels_file = labels_file[:num_samples]
    return labels, wavs_file, labels_file

def load_levi_hifi(data_dir, meta_csv, num_samples = None):
    df = pd.read_csv(meta_csv)
    df['wav'] = df['wav'].str.replace('$DATAROOT/LEVIorig11_HiFi', data_dir)
    wavs_file = df['wav'].to_list()
    labels = df['transcript'].to_list()
    extra = df.to_dict(orient='list')
    del extra['wav']
    del extra['transcript']
    if num_samples:
        labels = labels[:num_samples]
        wavs_file = wavs_file[:num_samples]
        extra_samples = {key: values[:num_samples] for key, values in extra.items()}
        extra  = extra_samples
    print('labels: ', labels)
    print('wavs_file: ', wavs_file)
    return labels, wavs_file, extra

def load_myst_json_flac(data_dir, json_path,split ='valid', num_samples = None):
    with open(os.path.join(json_path,split +'.json'), 'r') as json_file:
        data = json.load(json_file)
    data = list(data.values())
    wavs_file =[]
    labels_file = []
    labels = []
    labels_dir = os.path.join(json_path, 'transcript', split)
    for i in range(len(data)):
        wavs_file.append(data[i]['wav'].replace('../corpora/myst-v0.4.2',data_dir))
        wavs_file[i] = wavs_file[i].replace('.wav','.flac')
        labels.append(data[i]['transcript'])
        label_file = os.path.join(labels_dir,os.path.basename(wavs_file[i]).replace('.flac', '.trn'))
        labels_file.append(label_file)
    if num_samples:
        labels = labels[:num_samples]
        wavs_file = wavs_file[:num_samples]
        labels_file = labels_file[:num_samples]
    return labels, wavs_file, {'label_files':labels_file}

def load_data_csv(data_dir,
                  meta_csv,
                  wavs_file_col, 
                  labels_col,
                  data_root, 
                  num_samples=None):
    df = pd.read_csv(meta_csv)
    df[wavs_file_col] = df[wavs_file_col].str.replace(data_root, data_dir)
    wavs_file = df[wavs_file_col].to_list()
    labels = df[labels_col].to_list()
    extra = df.to_dict(orient='list')
    del extra[wavs_file_col]
    del extra[labels_col]
    if num_samples:
        labels = labels[:num_samples]
        wavs_file = wavs_file[:num_samples]
        extra_samples = {key: values[:num_samples] for key, values in extra.items()}
        extra  = extra_samples
    return labels, wavs_file, extra


def load_levi_lofi_5(data_dir, meta_csv, num_samples = None):
    df = pd.read_csv(meta_csv)
    df['wavs_file'] = df['wavs_file'].str.replace('$DATAROOT/first11', data_dir)
    wavs_file = df['wavs_file'].to_list()
    labels = df['transcript'].to_list()
    extra = df.to_dict(orient='list')
    del extra['wav']
    del extra['transcript']
    if num_samples:
        labels = labels[:num_samples]
        wavs_file = wavs_file[:num_samples]
        extra_samples = {key: values[:num_samples] for key, values in extra.items()}
        extra  = extra_samples
    print('labels: ', labels)
    print('wavs_file: ', wavs_file)
    return labels, wavs_file, extra

def load_dataset(dataset_name, **kwargs):
    print('kwargs: ', kwargs)
    datasets_csv_format = ['levi_lofi_5','levi_hifi','isat','myst_csv']
    if dataset_name == 'myst':
        return load_myst_original(data_dir = kwargs['data_dir'],
                         split = kwargs['split'],
                         num_samples = kwargs['num_samples'])
    elif dataset_name == 'myst_json':
        return load_myst_json_flac(data_dir = kwargs['data_dir'],
                         json_path = kwargs['json_path'],
                         split = kwargs['split'],
                         num_samples = kwargs['num_samples'])
    elif dataset_name in datasets_csv_format:
        return load_data_csv(data_dir =  kwargs['data_dir'],
                            meta_csv = kwargs['meta_csv'],    
                            wavs_file_col = kwargs['wavs_file_col'],
                            labels_col =  kwargs['labels_col'], 
                            data_root = kwargs['data_root'],
                            num_samples = kwargs['num_samples'])