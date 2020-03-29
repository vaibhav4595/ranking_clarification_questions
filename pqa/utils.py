import os
import sys
import math
import random
import pickle
import torch
from build_vocab import VocabEntry
from pdb import set_trace as bp

def get_ids_from_file(filepath):

    print("Getting Ids for ", filepath)

    id_set = set()
    fp = open(filepath)
    for line in fp:
        id_set.add(line.strip())

    return id_set

def get_post_dict(filepath, post_dict):

    # Hardcode maximum post length to 300
    print("Getting Content of Posts for", filepath)

    lines = open(filepath).readlines()[1:]
    for line in lines:
        line = line.strip().split('\t', 1)
        post_dict[line[0]] = line[1].split()[0:300]

def get_qa_data(filepath, qa_dict):

    # Hardcode maximum post length to 40
    print("Getting the top 10 retrieved question answer tuples for ", filepath) 

    lines = open(filepath).readlines()[1:]
    for line in lines:
        line = line.strip().split('\t')
        for i in range(1, len(line)):
            line[i] = line[i].split()[0:40]
        qa_dict[line[0]] = (line[1:11], line[11:])

def get_training_content(args):

    dirs = ['askubuntu.com', 'superuser.com', 'unix.stackexchange.com']

    train_ids = set()
    val_ids = set()

    for direc in dirs:
        train_id_path = os.path.join(os.path.join(args.data_path, direc), 'train_ids')
        temp_train_ids = get_ids_from_file(train_id_path)
        val_id_path = os.path.join(os.path.join(args.data_path, direc), 'tune_ids')
        temp_val_ids = get_ids_from_file(val_id_path)
        train_ids = train_ids | temp_train_ids
        val_ids = val_ids | temp_val_ids

    post_content = dict()

    for direc in dirs:
        post_content_path = os.path.join(os.path.join(args.data_path, direc), 'post_data.tsv')
        get_post_dict(post_content_path, post_content)

    qa_dict = {}
    for direc in dirs:
        qa_content_path = os.path.join(os.path.join(args.data_path, direc), 'qa_data.tsv')
        get_qa_data(qa_content_path, qa_dict)


    return train_ids, val_ids, post_content, qa_dict

def get_vocab(filename):

    print("Loading Vocabulary")
    fp = open(filename, 'rb')
    vocab = pickle.load(fp)

    return vocab

def batch_iter(ids, post_content, qa_dict, vocab, batch_size, shuffle=False):

    example_ids = []
    post_words = []
    question_words = []
    answer_words = []
    labels = []

    shuffled_ids = list(ids)
    if shuffle == True:
        random.shuffle(shuffled_ids)

    for idx in shuffled_ids:
        post = vocab.words2indices(post_content[idx])
        for i, (question, answer) in enumerate(zip(qa_dict[idx][0], qa_dict[idx][1])):
            if i == 0:
                labels.append(1)
            else:
                labels.append(0)

            example_ids.append(idx + '_' + str(i + 1))
            post_words.append(post)
            question_words.append(vocab.words2indices(question))
            answer_words.append(vocab.words2indices(answer))

    print("Total examples = ", len(example_ids))

    batch_num = math.ceil(len(example_ids) / batch_size)
    index_array = list(range(len(example_ids)))

    if shuffle == True:
        random.shuffle(index_array)

    for i in range(batch_num):

        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_ids = [example_ids[idx] for idx in indices]
        batch_posts = [post_words[idx] for idx in indices]
        batch_questions = [question_words[idx] for idx in indices]
        batch_answers = [answer_words[idx] for idx in indices]
        batch_labels = [labels[idx] for idx in indices]

        yield batch_ids, batch_posts, batch_questions, batch_answers, batch_labels

def pad_sequence(device, lister):

    input_pads = []
    max_length = 0
    for each in lister:
        list_len = len(each)
        max_length = max(max_length, list_len)
        pads = [1 for _ in range(list_len)]
        input_pads.append(pads)

    for i in range(len(lister)):
        pad_len = max_length - len(lister[i])
        lister[i] = lister[i] + [0] * pad_len
        input_pads[i] = input_pads[i] + [0] * pad_len

    return torch.tensor(lister).to(device=device), torch.tensor(input_pads).to(device=device)

def get_test_ranking(base_type, filepath, test_dict):

    lines = open(filepath).readlines()
    lines = [line.strip().split('\t') for line in lines]
    for each in lines:
        annot1 = each[0].split(',')
        annot2 = each[1].split(',')
        idx = base_type + '_' + annot1[1]
        best_ques = set([int(annot1[2]), int(annot2[2])])
        valid1 = set([int(idx) for idx in annot1[3].split()])
        valid2 = set([int(idx) for idx in annot2[3].split()])
        valid_ques = valid1 & valid2
        test_dict[idx] = {}
        test_dict[idx]['best'] = best_ques
        test_dict[idx]['valid'] = valid_ques
