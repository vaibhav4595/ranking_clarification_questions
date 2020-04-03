import os
import sys
import json
import argparse
import time
from utils import *
from pdb import set_trace as bp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../data/')

    args = parser.parse_args()

    test_ids = set()
    dirs = ['askubuntu.com', 'superuser.com', 'unix.stackexchange.com']
    for direc in dirs:
        test_id_path = os.path.join(os.path.join(args.data_path, direc), 'test_human_eval_ids')
        temp_ids = get_ids_from_file(test_id_path)
        test_ids = test_ids | temp_ids

    basic_names = ['askubuntu', 'superuser', 'unix']
    ground_truth = dict()
    for direc, base_name in zip(dirs, basic_names):
        human_annot_path = os.path.join(os.path.join(args.data_path, direc), 'human_annotations')
        get_test_ranking(base_name, human_annot_path, ground_truth)

    post_data = dict()
    for direc in dirs:
        post_path = os.path.join(os.path.join(args.data_path, direc), 'post_data.tsv')
        temp_ids_path = get_post_dict(post_path, post_data)

    qa_data = dict()
    for direc in dirs:
        qa_path = os.path.join(os.path.join(args.data_path, direc), 'qa_data.tsv')
        get_qa_data(qa_path, qa_data)

    print("Total test ids = ", len(ground_truth))

    filenames = ['ranked_list_evpi.json','ranked_list_pqa.json', 'ranked_list_bidi.json']

    data = []
    for each in filenames:
        data.append(json.loads(open(each).readlines()[0]))

    for each in data:
        pat = dict()
        for idx in each:
            if each[idx][0] in ground_truth[idx]['valid']:
                pat[idx] = 1
        new_pat = dict()
        for idx in pat:
            new_pat[idx] = (pat[idx], len(post_data[idx]))
        bins = [0] * 7
        for idx in new_pat:
            if new_pat[idx][0] == 1:
                bins[min(new_pat[idx][1] // 40, 6)] += 1
        print(bins)
