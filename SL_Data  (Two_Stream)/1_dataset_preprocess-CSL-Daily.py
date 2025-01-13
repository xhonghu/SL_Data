import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def csv2dict(anno_path):
    with open(anno_path,'r', encoding='utf-8') as f:
        inputs_list = f.readlines()
    info_dict = dict()
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list[1:]), total=len(inputs_list)-1):  # Exclude first line
        index, name, length, gloss, char, word, postag = file_info.strip().split("|")
        word = word.replace(" ", "")
        info_dict[file_idx] = {
            'fileid': name,
            'folder': name+'/*.jpg',
            'signer': 'unknown',
            'label': gloss,
            'text': word,
            'num_frames': length,
        }
    return info_dict

def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='preprocess/CSL-Daily/',
                        help='save prefix')
    parser.add_argument('--annotation-file', type=str, default='annotations/csl-daily/video_map.txt',
                        help='annotation file')
    parser.add_argument('--split-file', type=str, default='annotations/csl-daily/split_1.txt',
                        help='split file')

    args = parser.parse_args()
    mode = ["train", "dev", "test"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    
    # generate information dict
    information = csv2dict(f"{args.annotation_file}")
    with open(f"{args.split_file}",'r', encoding='utf-8') as f:
        files_list = f.readlines()  
    train_files = []
    dev_files = []
    test_files = []
    for file_idx, file_info in tqdm(enumerate(files_list[1:]), total=len(files_list)-1):  # Exclude first line
        name, split = file_info.strip().split("|")  
        if split == 'train':
            train_files.append(name)
        elif split == 'dev':
            dev_files.append(name)
        elif split == 'test':
            test_files.append(name)
    assert len(train_files) + len(dev_files) + len(test_files) == len(information)
    information_pack = dict()
    for md in mode:
        information_pack[md] = dict()
    train_id = 0
    dev_id = 0
    test_id = 0
    for info_key, info_data in information.items():
        if info_data['fileid'] in train_files:
            information_pack['train'][train_id] = info_data
            train_id += 1
        elif info_data['fileid'] in dev_files:
            information_pack['dev'][dev_id] = info_data
            dev_id +=1
        elif info_data['fileid'] in test_files:
            information_pack['test'][test_id] = info_data
            test_id +=1
        else:
            information_pack['train'][train_id] = info_data    #S000007_P0003_T00
            train_id += 1
    assert len(information_pack['train']) + len(information_pack['dev']) + len(information_pack['test']) == len(information)

    for md in mode:
        np.save(f"./{args.dataset}/{md}_info.npy", information_pack[md])
        # generate groudtruth stm for evaluation
    # update the total gloss dict
    sign_dict_update(sign_dict, information)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
