import numpy as np
import torch
import torch.nn.functional as F
from pdb import set_trace as bp

class EVPI(torch.nn.Module):

    def __init__(self, args, vocab):
        super(EVPI, self).__init__()

        self.vocab = vocab
        self.args = args
        self.embed_size = args.embed_size
        self.embedding = torch.nn.Embedding(num_embeddings=len(vocab),\
                                            embedding_dim=self.embed_size,\
                                            padding_idx=0)

        self.question_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           batch_first=True)
        self.answer_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           batch_first=True)
 
        self.post_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           batch_first=True)

        answer_linear = []
        answer_linear.append(torch.nn.Linear(2 * self.args.lstm_hidden_size, self.args.feedforward_hidden))
        for i in range(self.args.linear_layers - 1):
            answer_linear.append(torch.nn.Linear(self.args.feedforward_hidden, self.args.feedforward_hidden))

        util_linear = []
        util_linear.append(torch.nn.Linear(3 * self.args.lstm_hidden_size, self.args.feedforward_hidden))
        for i in range(self.args.linear_layers - 1):
            util_linear.append(torch.nn.Linear(self.args.feedforward_hidden, self.args.feedforward_hidden))

        self.answer_linear = torch.nn.ModuleList(answer_linear)
        self.util_linear = torch.nn.ModuleList(util_linear)
        self.class_layer = torch.nn.Linear(self.args.feedforward_hidden, 2)

        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(p=args.dropout)

    def load_vector(self, args, vocab):

        print("Loading Pretrained Vectors")
        vectors = np.zeros((len(vocab), args.embed_size))
        
        for word in vocab.word2id:
            idx = vocab[word]
            vec = vocab.word2vec[idx]
            vectors[idx] = vec

        vectors = np.asarray(vectors)
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        if args.fine_tune == 0:
            self.embedding.weight.requires_grad = False

    def forward(self, ids, post_tuple, question_tuple, answer_tuple):

        post_ids, post_pad_idx = post_tuple
        question_ids, question_pad_idx = question_tuple
        answer_ids, answer_pad_idx = answer_tuple

        selection_idx = []
        for i, idx in enumerate(ids):
            if idx.endswith('1'):
                selection_idx.append(i)

        post_ids = post_ids[selection_idx, ]
        post_pad_idx = post_pad_idx[selection_idx, ]

        post_embed = self.embedding(post_ids)
        question_embed = self.embedding(question_ids)
        answer_embed = self.embedding(answer_ids)

        post_lengths = post_pad_idx.sum(dim=1)
        question_lengths = question_pad_idx.sum(dim=1)
        answer_lengths = answer_pad_idx.sum(dim=1)

        packed_post = torch.nn.utils.rnn.pack_padded_sequence(post_embed,\
                                                              post_lengths,\
                                                              batch_first=True,\
                                                              enforce_sorted=False)

        packed_question = torch.nn.utils.rnn.pack_padded_sequence(question_embed,\
                                                              question_lengths,\
                                                              batch_first=True,\
                                                              enforce_sorted=False)

        packed_answer = torch.nn.utils.rnn.pack_padded_sequence(answer_embed,\
                                                              answer_lengths,\
                                                              batch_first=True,\
                                                              enforce_sorted=False)

        post_hiddens, _ = self.post_lstm(packed_post)
        question_hiddens, _ = self.question_lstm(packed_question)
        answer_hiddens, _ = self.answer_lstm(packed_answer)

        post_hiddens, _ =  torch.nn.utils.rnn.pad_packed_sequence(post_hiddens,\
                                                               batch_first=True)
        question_hiddens, _ =  torch.nn.utils.rnn.pad_packed_sequence(question_hiddens,\
                                                               batch_first=True)
        answer_hiddens, _ =  torch.nn.utils.rnn.pad_packed_sequence(answer_hiddens,\
                                                               batch_first=True)

        post_vector = post_hiddens.sum(dim=1) / post_lengths.unsqueeze(1)
        question_vector = question_hiddens.sum(dim=1) / question_lengths.unsqueeze(1)
        answer_vector = answer_hiddens.sum(dim=1) / answer_lengths.unsqueeze(1)

        new_post_vector = torch.repeat_interleave(post_vector, 10, dim=0)
        pqa_vector = torch.cat([new_post_vector, question_vector, answer_vector], dim=1)
        for i in range(self.args.linear_layers):
            pqa_vector = self.dropout(pqa_vector)
            pqa_vector = self.relu(self.util_linear[i](pqa_vector))
        pqa_probs = self.class_layer(pqa_vector)

        #selection_idx = []
        #for i, idx in enumerate(ids):
        #    if idx.endswith('1'):
        #        selection_idx.append(i)

        #selection_idx = torch.tensor(selection_idx).to(device=self.args.device)
        # p(qi | p)
        # select the qi and p
        former_q_evpi = question_vector[selection_idx,]
        #former_q_evpi = torch.index_select(question_vector, dim=0, index=selection_idx)
        #latter_p_evpi = torch.index_select(post_vector, dim=0, index=selection_idx)

        #pq_vector = torch.cat([former_q_evpi, latter_p_evpi], dim=1)
        pq_vector = torch.cat([post_vector, former_q_evpi], dim=1)
        for i in range(self.args.linear_layers):
            pq_vector = self.dropout(pq_vector)
            pq_vector = self.relu(self.answer_linear[i](pq_vector))

        return pq_vector, pqa_probs
