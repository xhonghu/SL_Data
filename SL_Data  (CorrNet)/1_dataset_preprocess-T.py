import os
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm

def csv2dict(anno_path, dataset_type):
    inputs_list = pandas.read_csv(anno_path)
    inputs_list = (inputs_list.to_dict()['name|video|start|end|speaker|orth|translation'].values())
    info_dict = dict()
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        name, video, start, end, speaker, orth, translation = file_info.split("|")
        info_dict[file_idx] = {
            'fileid': name,
            'folder': f"{dataset_type}/{name}"+"/*.png",
            'signer': speaker,
            'label': orth,
            'text': translation + ' .',
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
    parser.add_argument('--dataset', type=str, default='preprocess/phoenix2014-T',
                        help='save prefix')
    parser.add_argument('--annotation-prefix', type=str, default='annotations/phoenix2014t/PHOENIX-2014-T.{}.corpus.csv',
                        help='annotation prefix')

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.annotation_prefix.format(md)}", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)