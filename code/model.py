import numpy as np
import torch
import torch.nn.functional as F
from pdb import set_trace as bp

class EVPI(torch.nn.Module):

    def __init__(self, args, vocab):
        super(CNN, self).__init__()

        self.args = args
        self.embed_size = args.embed_size
        self.embedding = torch.nn.Embedding(num_embeddings=len(vocab),\
                                            embedding_dim=self.embed_size,\
                                            padding_idx=0)

        self.question_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.hidden_size,\
                                           batch_first=True)
        self.answer_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.hidden_size,\
                                           batch_first=True)
 
        self.post_lstm = torch.nn.LSTM(input_size=self.embed_size,\
                                           hidden_size=self.args.hidden_size,\
                                           batch_first=True)

        answer_linear = []
        for i in range(self.args.linear_layers):
            answer_linear.append(torch.nn.Linear(self.args.hidden_size, self.args.feedforward_hidden))

        util_linear = []
        for i in range(self.args.linear_layers):
            util_linear.append(torch.nn.Linear(self.args.hidden_size, self.args.feedforward_hidden))

        self.answer_linear = torch.nn.ModuleList(answer_linear)
        self.util_linear = torch.nn.ModuleList(util_linear)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=args.dropout)

    def load_vector(self, args, vocab):

        print("Loading Pretrained Vectors")
        vectors = np.zeros((len(vocab), args.embed_size))
        
        fp = open(args.embed_file)

        for word in vocab.word2id:
            idx = vocab[word]
            vec = vocab.word2vec[idx]
            vectors[idx] = vec

        vectors = np.asarray(vectors)
        self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        if args.fine_tune == 0:
            self.embedding.weight.requires_grad = False

    def forward(self):

        # TODO: Implememt the EVPI Procedure
        pass
