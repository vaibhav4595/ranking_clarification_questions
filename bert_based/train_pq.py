import os
import sys
import pickle
import argparse
import numpy as np
import torch
import time
import utils
from bert_pq import BertRank
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from pdb import set_trace as bp

def train():

    device = args.device

    print("Loading Bert and creating model")
    model = BertRank(args).to(device=device)

    no_decay = ["bias", "LayerNorm.weight"]

    parameters = list(model.parameters())
    # referenced from https://github.com/huggingface/transformers/blob/master/examples/run_xnli.py
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,\
                      weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer,\
                                            num_warmup_steps = 1920,\
                                            num_training_steps = 192000)

    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    total_loss = 0 
    total_steps = 384000
    model.train()
    iterations = 1 
    start_time = time.time()

    print("Beginning Training")
    model.zero_grad()

    flag = False
    for ep in range(args.max_epochs):

        for batch in utils.bert_batch_iter(train_ids, post_content, qa_dict, args.batch_size):

            ids, posts, questions, _, labels = batch

            outputs = model(ids, posts, questions)

            labels = torch.tensor(labels).to(device=device)

            loss = criterion(outputs, labels)
            loss = loss / args.accumulation_size
            total_loss += loss.item()
            loss.backward()

            # accumulate to avoid OOM issues
            if iterations % args.accumulation_size == 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                print("Iteration %d, Loss %f, Elapsed %s" %(iterations,\
                                                  total_loss,\
                                                  utils.format_time(time.time() - start_time)))
                total_loss = 0 

            iterations += 1

            if iterations > total_steps:
                flag = True
                break

        modelpath = os.path.join(args.model_dir, "bert_epoch_" + str(ep) + ".pkl")
        torch.save(model, modelpath)
        if flag == True:
            break

    modelpath = os.path.join(args.model_dir, "bert_final.pkl") 
    torch.save(model, modelpath)
    print("Training Ended")

if __name__ == '__main__':

   
    # TOTAL TRAIN EXAMPLES = 616810 (61681 examples * 10 retrived)
     
    parser = argparse.ArgumentParser()

    #parser.add_argument('--mode', type=str, required=True)
    parser.add_argument("--data_path", type=str, default="../../../datasets/clarification_questions_dataset/data/")
    parser.add_argument("--fine_tune", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="../models/bert_pq/base/")
    parser.add_argument("--max_ans_len", type=int, default=40)
    parser.add_argument("--max_ques_len", type=int, default=40)
    parser.add_argument("--max_post_len", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--valid_iter", type=int, default=2500)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_num_trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss_parameter", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_steps", type=int, default=800000)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--accumulation_size", type=int, default=2)
    parser.add_argument("--bert_type", type=str, default="base") 
    args = parser.parse_args()

    print("Preparing Training and Validation Sets for Training")

    train_ids, val_ids, post_content, qa_dict = utils.get_training_content(args)
    print("Total examples = ", len(train_ids))
    train()
