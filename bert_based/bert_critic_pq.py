import torch
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import XLNetTokenizer, XLNetModel

from pdb import set_trace as bp

class BertRank(torch.nn.Module):

    def __init__(self, args):

        super(BertRank, self).__init__()

        self.args = args

        if self.args.bert_type == "base":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                do_lower_case=True)
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if self.args.bert_type == "distil":
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', \
                do_lower_case=True)
            self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')

        if self.args.bert_type == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.bert = XLNetModel.from_pretrained('xlnet-base-cased')

        self.class_layer = torch.nn.Linear(768, 1)

        self.dropout = torch.nn.Dropout(args.dropout)

    def get_input(self, posts, questions):
        lengths = []
        max_length = 0
        t_posts = []
        t_questions = []

        for p, q in zip(posts, questions):
            p = ['[CLS'] + self.tokenizer.tokenize(" ".join(p))[:self.args.max_post_len] + ['[SEP]']
            q = self.tokenizer.tokenize(" ".join(q))[:self.args.max_ques_len] + ['[SEP]']
            len_p, len_q = len(p), len(q)
            max_length = max(max_length, len_p + len_q)
            lengths.append((len_p, len_q))
            t_posts.append(p)
            t_questions.append(q)

        indexed_examples = []
        seg_mask = []
        attn_mask = []

        for p, q, lens in zip(t_posts, t_questions, lengths):

            temp_seg = [0] * lens[0]
            temp_seg = temp_seg + [1] * lens[1]
            temp_seg = temp_seg + [1] * (max_length - sum(lens))

            final_example = p + q
            final_example = final_example + ['[PAD]'] * (max_length - sum(lens))
            final_example = self.tokenizer.convert_tokens_to_ids(final_example)
            attn = [int(t_id > 0) for t_id in final_example]
            indexed_examples.append(final_example)
            seg_mask.append(temp_seg)
            attn_mask.append(attn)

        indexed_examples = torch.tensor(indexed_examples).to(self.args.device)
        seg_mask = torch.tensor(seg_mask).to(self.args.device)
        attn_mask = torch.tensor(attn_mask).to(self.args.device)

        return indexed_examples, seg_mask, attn_mask

    def forward(self, ids, posts, questions):

        inputs, seg_mask, attn_mask = self.get_input(posts, questions)
        if self.args.bert_type == "base":
            outputs = self.bert(inputs, attention_mask=attn_mask,\
                token_type_ids=seg_mask)
        if self.args.bert_type == "distil":
            outputs = self.bert(inputs, attention_mask=attn_mask)
        if self.args.bert_type == "xlnet":
            outputs = self.bert(inputs, attention_mask=attn_mask,\
                token_type_ids=seg_mask)

        cls_output = outputs[0][:, 0]
        output = self.class_layer(cls_output)
 
        return torch.sigmoid(output)
