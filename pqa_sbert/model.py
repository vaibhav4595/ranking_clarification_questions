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

        self.bidi = False
        if self.args.bidirectional == 1:
            self.bidi = True

        self.question_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           bidirectional=self.bidi,\
                                           batch_first=True)
        self.answer_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           bidirectional=self.bidi,\
                                           batch_first=True)
 
        self.post_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.lstm_hidden_size,\
                                           bidirectional=self.bidi,\
                                           batch_first=True)

        factor = 1
        if self.bidi == True:
            factor = 2
        util_linear = []
        util_linear.append(torch.nn.Linear(2 * factor * 768, self.args.feedforward_hidden)) # new model

        for i in range(self.args.linear_layers - 1):
            util_linear.append(torch.nn.Linear(self.args.feedforward_hidden, self.args.feedforward_hidden))

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

    def forward(self, posts, questions, answers):

        #pqa_vector = torch.ones(600).cuda()  # torch.cat([post_vector, question_vector, answer_vector], dim=1)
        pqa_vector = torch.cat([posts, questions], dim=1) #

        for i in range(self.args.linear_layers):

            pqa_vector = self.dropout(pqa_vector)
            pqa_vector = self.relu(self.util_linear[i](pqa_vector))

        pqa_probs = self.class_layer(pqa_vector)

        return pqa_probs
