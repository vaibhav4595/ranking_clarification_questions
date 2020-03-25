import os
import sys
import pickle
import numpy as np
import torch
import argparse
import time
from prev_model import EVPI
from build_vocab import VocabEntry
from utils import *
from pdb import set_trace as bp


def test():

    print("Loading Model")
    model = torch.load(args.model_path)
    model.eval()
    device="cuda"

    best_precision = [0 for _ in range(10)]
    valid_precision = [0 for _ in range(10)]
    total_exps = 0

    softmax = torch.nn.Softmax(dim=1)
    cosine_sim = torch.nn.functional.cosine_similarity

    for idx in ground_truth:

        total_exps += 1

        post = post_data[idx]
        q_candidate = qa_data[idx][0]
        ans_candidate = qa_data[idx][1]

        expanded_posts = [[each for each in post] for _ in range(10)]

        q_words = vocab.words2indices(q_candidate)
        ans_words = vocab.words2indices(ans_candidate)

        #q_words, q_pads = pad_sequence(device, q_words)
        #ans_words, ans_pads = pad_sequence(device, ans_words)

        post_words = vocab.words2indices(expanded_posts)
        post_words, post_pads = pad_sequence(device, post_words)

        evpi_scores = []
        for i, qs in enumerate(q_candidate):

            current_q = [each for each in q_words[i]]
            current_ans = [each for each in ans_words[i]]

            temp_idxs = []
            temp_idxs.append(idx + '_1')
            counter = 2
            qs = []
            ans = []
            qs.append(current_q)
            ans.append(current_ans)

            for j in range(10):
                if i == j:
                    continue
                temp_idxs.append(idx + '_' + str(counter))
                counter += 1
                qs.append([each for each in q_words[j]])
                ans.append([each for each in ans_words[j]])

            ans_average_vector = vocab.id2vector(ans)

            qs, qs_pads = pad_sequence(device, qs)
            ans, ans_pads = pad_sequence(device, ans)
 
            post_tuple = (post_words, post_pads)
            qs_tuple = (qs, qs_pads)
            ans_tuple = (ans, ans_pads)
 
            pq_rep, pqa_prob = model(temp_idxs, post_tuple, qs_tuple, ans_tuple)
            util_probs = softmax(pqa_prob)[:, 1]

            ans_avg_vector = torch.from_numpy(np.asarray(ans_average_vector)).to(device=device)
            pq_rep = torch.repeat_interleave(pq_rep, 10, dim=0)
            answer_model = 1 - cosine_sim(pq_rep, ans_avg_vector)
            answer_prob = torch.exp(-answer_model)

            score = torch.sum(answer_prob * util_probs)
            evpi_scores.append((score.item(), i))

        ranked_list = sorted(evpi_scores, key=lambda x:x[0], reverse=True)
        ranked_list = [each[1] for each in ranked_list]

        temp_best_prec = [0 for _ in range(10)]
        temp_valid_prec = [0 for _ in range(10)]
        for i, value in enumerate(ranked_list):
            if value in ground_truth[idx]['best']:
                temp_best_prec[i] = 1
            if value in ground_truth[idx]['valid']:
                temp_valid_prec[i] = 1

        for i in range(1, 10):
            temp_best_prec[i] = (temp_best_prec[i] + temp_best_prec[i - 1]) / (i)
            temp_valid_prec[i] = (temp_valid_prec[i] + temp_valid_prec[i - 1]) / (i)

        for i in range(10):
            best_precision[i] += temp_best_prec[i]
            valid_precision[i] += temp_valid_prec[i]

    for i in range(10):
        best_precision[i] /= total_exps
        valid_precision[i] /= total_exps

    print("Best Precision Scores", best_precision)
    print("Valid Precision Scores", valid_precision)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='../models/evpi_mod/model.pkl')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')

    args = parser.parse_args()

    vocab = get_vocab(args.vocab_file)

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

    print("Beginning Testing")
    test()
