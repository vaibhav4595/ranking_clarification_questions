import os
import sys
import pickle
import argparse
import numpy as np
import torch
import time
from model import EVPI
from build_vocab import VocabEntry
from utils import *
from pdb import set_trace as bp
from info_nce_loss import InfoNCELoss

def train():

    device = args.device

    log_every = args.log_every
    valid_iter = args.valid_iter
    train_iter = 0
    cum_loss = 0
    avg_loss = 0
    avg_util_loss = 0
    avg_answer_loss = 0
    valid_num = 0
    patience = 0
    num_trial = 0
    hist_valid_scores = []
    begin_time = time.time()

    vocab = get_vocab(args.vocab_file)

    model = EVPI(args, vocab)

    if args.use_embed == 1:
       model.load_vector(args, vocab)

    print("Placing model on ", args.device)
    if args.device == 'cuda':
       model.cuda()

    lr = args.lr
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)

    # The loss functions
    #criterion = torch.nn.CrossEntropyLoss().to(device=device)
    #criterion = FocalLoss()
    criterion = InfoNCELoss().cuda()

    print("Beginning Training")
    model.train()

    cosine_function = torch.nn.functional.cosine_similarity

    model_counter = 0
    train_iter = 0
    for ep in range(args.max_epochs):

        val_iter = 0

        count = 0
        hello = set()
        for ids, posts, questions, answers, labels in batch_iter(train_ids, \
                            post_content, qa_dict, vocab, args.batch_size, shuffle=False):

            train_iter += 1

            optim.zero_grad()

            question_vectors = vocab.id2vector(questions)
            post_vectors = vocab.id2vector(posts)
            answer_vectors = vocab.id2vector(answers)

            padded_posts, post_pad_idx = pad_sequence(args.device, posts)
            padded_questions, question_pad_idx = pad_sequence(args.device, questions)
            padded_answers, answer_pad_idx = pad_sequence(args.device, answers)

            pqa_probs = model(ids, (padded_posts, post_pad_idx),\
                      (padded_questions, question_pad_idx),\
                      (padded_answers, answer_pad_idx))

            labels = torch.tensor(labels).to(device=args.device)
            total_loss = criterion(pqa_probs, labels)

            #bp()

            avg_loss += total_loss.item()
            cum_loss += total_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.clip_grad)
            optim.step()

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg.loss %.6f, time elapsed %.2f'\
                     % (ep + 1, train_iter, avg_loss / log_every, time.time() - begin_time), file=sys.stderr)

                begin_time = time.time()
                avg_loss = 0

            if train_iter % valid_iter == 0:

                print('epoch %d, iter %d, cum.loss %.2f, time elapsed %.2f'\
                     % (ep + 1, train_iter, cum_loss / valid_iter, time.time() - begin_time), file=sys.stderr)

                cum_loss = 0
                valid_num += 1

                print("Begin Validation ", file=sys.stderr)

                model.eval()

                val_loss = get_val_loss(vocab, args, model)
                model.train()

                print('validation: iter %d, loss %f' % (train_iter, val_loss), file=sys.stderr)

                is_better = (len(hist_valid_scores) == 0) or (val_loss < min(hist_valid_scores))
                hist_valid_scores.append(val_loss)

                if is_better:
                    patience = 0
                    print("Save the current model and optimiser state")
                    torch.save(model, args.model_save_path)
                    #torch.save(model, args.model_save_path + '.' + str(val_loss) + '-' + str(model_counter))
                    #model_counter += 1
                    torch.save(optim.state_dict(), args.model_save_path + '.optim')

                elif patience < args.patience:

                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == args.max_num_trials:
                            print('early stop!', file=sys.stderr)
                            return

                        lr = lr * args.lr_decay

                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                        model = load(args.model_save_path)
                        model.train()

                        print('restore parameters of the optimizers', file=sys.stderr)

                        optim = torch.optim.Adam(list(model.parameters()), lr=lr)
                        optim.load_state_dict(torch.load(args.model_save_path + '.optim'))
                        for state in optim.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(args.device)
                        for group in optim.param_groups:
                            group['lr'] = lr

                        patience = 0
    print("Training Finished", file=sys.stderr) 


def get_val_loss(vocab, args, model):

    total_loss = 0
    total_util_loss = 0
    total_answer_loss = 0

    util_examples = 0
    answer_examples = 0

    model.eval()

    #criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = InfoNCELoss(size_average=False).cuda()

    for ids, posts, questions, answers, labels in batch_iter(val_ids, \
                       post_content, qa_dict, vocab, args.batch_size, shuffle=False):

       
        util_examples += len(ids)

        question_vectors = vocab.id2vector(questions)
        post_vectors = vocab.id2vector(posts)
        answer_vectors = vocab.id2vector(answers)

        padded_posts, post_pad_idx = pad_sequence(args.device, posts)
        padded_questions, question_pad_idx = pad_sequence(args.device, questions)
        padded_answers, answer_pad_idx = pad_sequence(args.device, answers)

        pqa_probs = model(ids, (padded_posts, post_pad_idx),\
                  (padded_questions, question_pad_idx),\
                  (padded_answers, answer_pad_idx))

        labels = torch.tensor(labels).to(device=args.device)
        util_loss = criterion(pqa_probs, labels)

        #bp()

        total_util_loss += util_loss.item()

    total_loss = (total_util_loss / util_examples)
    model.train()

    return total_loss

def load(model_path):

    model = torch.load(model_path)
    model.question_lstm.flatten_parameters()
    model.answer_lstm.flatten_parameters()
    model.post_lstm.flatten_parameters()
    return model

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    #parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_embed', type=int, default=1)
    parser.add_argument('--fine_tune', type=int, default=1)
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--model_path', type=str, default='../models/pqa/model.pkl')
    parser.add_argument('--model_save_path', type=str, default='../models/pqa/model.pkl')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--max_ans_len', type=int, default=40)
    parser.add_argument('--max_ques_len', type=int, default=40)
    parser.add_argument('--max_post_len', type=int, default=300)
    parser.add_argument('--lstm_hidden_size', type=int, default=200)
    parser.add_argument('--feedforward_hidden', type=int, default=200)
    parser.add_argument('--bidirectional', type=int, default=0)
    parser.add_argument('--linear_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--valid_iter', type=int, default=2500)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.25)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_num_trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--loss_parameter", type=float, default=0.2)
    parser.add_argument("--only_last", type=int, default=0)

    args = parser.parse_args()

    print("Preparing Training and Validation Sets for Training")

    train_ids, val_ids, post_content, qa_dict = get_training_content(args)

    train()
