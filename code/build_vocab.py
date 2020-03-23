import sys
import json
import pickle
import argparse
import numpy as np
from pdb import set_trace as bp

class VocabEntry(object):
    def __init__(self, vocab_type='word'):
        self.word2id = dict()
        self.word2vec = dict()
        self.vocab_type = vocab_type
        self.unk_id = 2

        self.pad_id = 0
        self.sep_id = 1

        self.word2id['<<!PAD!>>'] = 0
        self.word2id['<<!SEP!>>'] = 1
        self.word2id['<<!UNK!>>'] = 2

        self.pad_tok = '<<!PAD!>>'
        self.sep_tok = '<<!SEP!>>'
        self.unk_tok = '<<!UNK!>>'

        # Hardcode the embedding size as 200, to obey the paper
        self.word2vec[0] = np.zeros((200, ))
        self.word2vec[1] = np.random.random((200, ))
        self.word2vec[2] = np.random.random((200, ))

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word, vec):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            self.word2vec[wid] = vec
            return wid
        else:
            return self[word]

    def numberize(self, sents):
        if self.vocab_type == 'word':
          return self.words2indices(sents)

    def denumberize(self, ids):
      if type(ids[0]) == list:
        if self.vocab_type == 'word':
          return [' '.join([self.id2word[w] for w in sent]) for sent in ids]
        else:
          return [''.join([self.id2word[w] for w in sent]) for sent in ids]
      else:
        if self.vocab_type == 'word':
          return ' '.join([self.id2word[w] for w in ids])
        else:
          return ''.join([self.id2word[w] for w in ids])

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def words2vectors(self, sents):
        if type(sents[0]) == list:
            return [[self.word2vec[self[w]] for w in s] for s in sents]
        else:
            return [self.word2vec[self[w]] for w in sents]

    def sentence2vector(self, sents):
        vectors = self.words2vectors(sents)
        if type(vectors[0]) == list:
            return [np.sum(vector, axis=0) / len(vector) for vector in vectors]
        else:
            return np.sum(vectors, axis=0) / len(vectors)
        
    def from_corpus(self, input_file):

        fp = open(args.input_file)
       
        for line in fp:
            word, vec = line.split(' ', 1)
            word = word.lower().strip()
            vec = np.fromstring(vec, sep=' ') 
            self.add(word, vec)

def test_vocab(vocab):


    assert vocab.unk_id == 2
    assert vocab.words2indices([vocab.pad_tok, vocab.pad_tok]) == [0, 0]
    assert vocab.words2indices([[vocab.pad_tok], [vocab.pad_tok]]) == [[0], [0]]
    assert (vocab.sentence2vector([[vocab.pad_tok], [vocab.pad_tok]])[0] == np.zeros((200, ))).all()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', default='vocab.txt',\
        help="Processed File containing a word in each line")
    parser.add_argument('--save_name', default='vocab.pkl')

    args = parser.parse_args()

    print("Creating Vocabulary")
    vocab = VocabEntry()
    vocab.from_corpus(args.input_file)

    print("Total Length of Vocab is ", len(vocab))
    print("Dumping the vocabulary")
    fp = open(args.save_name, 'wb')
    pickle.dump(vocab, fp)
    fp.close()

    test_vocab(vocab)
