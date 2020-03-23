import numpy as np
import torch
import random
from torch.utils import data
import json
import h5py

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def word_tokenize(s):
  sent = s.lower()
  sent = re.sub('[^A-Za-z0-9\s]+',' ', sent)
  return sent.split()

def sentences_to_words(sentences):
  words = []
  for s in sentences:
    words.extend(word_tokenize(str(s.lower())))
  return words

# Due to the nature of propose and rank framework -> have to create separate dataloaders
# So, basically in evaluation code -> once the loop has ended -> just group by video id
# So, if you store predictions by video id, it should be fine?
#
#

class Dataset(data.Dataset):
  'Characterizes a very Basic Random Video dataset'
  def __init__(self, partition, vocab):
        'Initialization: Video Id == Sentence ID uniquely determines a data point'
        self.annotation_IDs = []
        self.annotation_to_video = {}
        self.annotation_to_times = {}
        self.annotation_to_text = {}
        self.annotation_to_time = {}
        self.vocab = vocab
        self.rgb_file = None
        #self.flow_file = '../data/didemos/average_global_flow.h5'
        self.flow_file = '../data/didemos/average_fc7.h5'
        self.video_to_flow = {}
        self.partition = partition

        with open(self.partition) as data_file:
            data = json.load(data_file)
            for row in data:
                self.annotation_IDs.append(row['annotation_id'])
                self.annotation_to_video[row['annotation_id']] = row['video']
                self.annotation_to_text[row['annotation_id']] = row['description']
                self.annotation_to_time[row['annotation_id']] = row['times']   ##[0]

        with h5py.File(self.flow_file, 'r') as f:
            for key in f.keys():
                data = list(f[key])
                self.video_to_flow[key] = np.asarray(data)

  def __len__(self):
        'Denotes the total number of video-sentence pairs'
        return len(self.annotation_IDs)

  def __getitem__(self, index):
        'Generates one sample of video-sentence pair'
        # Select annotation -> then go the video -> and get the corresponding clips
        # The feature for a video is 6 x 1024 -> this means it is clip-wise, why is it clip wise?
        video_id_pos = self.annotation_to_video[self.annotation_IDs[index]]
        video_time_pos = self.annotation_to_time[self.annotation_IDs[index]]
        # No matter what the time duration is, you will anyways mean pool things here
        # Based on this mean pool, there is a global feature, there is a local feature andt there are timepoint indices
        # Right now, I can just take send one frame
        # print(video_time)
        # print("Shape of the Video data = {}".format(np.asarray(self.video_to_flow[video_id_pos][int(video_time_pos[0]):int(video_time_pos[1])+1, :]).shape))
        time_index_pos = random.randint(0,3)
        #print(time_index_pos)
        #print(video_time_pos)
        local_features_pos = np.mean(self.video_to_flow[video_id_pos][int(video_time_pos[time_index_pos][0]):int(video_time_pos[time_index_pos][1])+1, :], axis=0) # This just returns the first part of the clip -> should mean pool for many
        global_features_pos = np.mean(self.video_to_flow[video_id_pos], axis=0)
        #print("Global - Local = {} {}".format(global_features_pos.shape, local_features_pos.shape))
        num_frames_pos = self.video_to_flow[video_id_pos].shape[0]
        # In data loader batch has to be constructed here
        sent_pos = self.vocab.src.numberize(self.annotation_to_text[self.annotation_IDs[index]])
        # Pad/Limit Sentences to a max limit -> see the dataset statistics -> paper says 30?
        #print("Real sentence length is = {}".format(len(sent_pos)))
        max_len = 75
        sent_pos = sent_pos[:max_len]
        sent_pos = sent_pos + [0]*(max_len - len(sent_pos))
        # Stats for Debugging
        #print("Video ID: ", video_id)
        #print("Sentence: ", np.asarray(sent).shape)
        #print("Features: ", np.asarray(features).shape)
        #print("Video Time: ", np.asarray(video_time).shape)

        ## Negative Sampling Code Starts Here
        ## this is controlled by Lambda
        sampling_lambda = 0.9
        sampling_random = random.uniform(0, 1)
        if sampling_random >= 1.9: # no inter video
            # Inter-Video Negative Sampling
            index_rand_id =  random.randint(0, len(self.annotation_IDs)-1)
            time_index_neg = random.randint(0, 3)
        else:
            # Intra-Video Negative Sampling
            index_rand_id = index #.deepcopy()
            possible =  [ (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 1), (4, 5), (3, 4), (2, 3), (1, 2), (3, 5), (0, 2), (2, 4), (2, 5), (1, 3), (1, 5), (0, 3), (1, 4), (0, 4) ]
            #print(self.annotation_to_time[self.annotation_IDs[index_rand_id]])
            #print(self.video_to_flow[self.annotation_to_video[self.annotation_IDs[index_rand_id]]])
            possible =  [ x for x in possible if x not in self.annotation_to_time[self.annotation_IDs[index_rand_id]] and x[1] <= self.video_to_flow[self.annotation_to_video[self.annotation_IDs[index_rand_id]]].shape[1] ]
            time_index_neg = random.choice(possible)[0]

        print("Pos = {}, Neg = {}, Num Frames = {}".format(np.asarray(video_time_pos)[:1].squeeze(), [time_index_neg, time_index_neg], num_frames_pos))

        video_id_neg = self.annotation_to_video[self.annotation_IDs[index_rand_id]]
        offset = random.randint(1, 2)
        # put data prior here
        # 70% -> 
        video_time_neg = [time_index_neg, time_index_neg] # +1 22% of the time
        # No matter what the time duration is, you will anyways mean pool things here
        # Based on this mean pool, there is a global feature, there is a local feature andt there are timepoint indices
        # Right now, I can just take send one frame
        # print(video_time)
        #print("Shape of the Video data = {}".format(np.asarray(self.video_to_flow[video_id]).shape))
        #time_index_neg = random.randint(0,5)
        #print(int(video_time_neg[time_index_neg][0]))
        #print(int(video_time_neg[time_index_neg][1]))
        #print(video_time_neg)
        local_features_neg = np.mean(self.video_to_flow[video_id_neg][int(video_time_neg[0]):int(video_time_neg[1])+1, :], axis=0) # This just returns the first part of the clip -> should mean pool for many
        global_features_neg = np.mean(self.video_to_flow[video_id_neg], axis=0)
        num_frames_neg = self.video_to_flow[video_id_neg].shape[1]
        # In data loader batch has to be constructed here
        sent_neg = self.vocab.src.numberize(self.annotation_to_text[self.annotation_IDs[index_rand_id]])
        # Pad/Limit Sentences to a max limit -> see the dataset statistics -> paper says 30?
        sent_neg = sent_neg[:max_len]
        sent_neg = sent_neg + [0]*(max_len - len(sent_neg))

        #print("Feature Sizes = {} {} {} {} {} {} {} {}".format(np.asarray(sent_pos).shape, np.asarray(local_features_pos).shape, np.asarray(global_features_pos).shape, np.asarray(video_time_pos).shape, np.asarray(sent_neg).shape, np.asarray(local_features_neg).shape, np.asarray(global_features_neg).shape, np.asarray(video_time_neg).shape))

        return torch.from_numpy(np.asarray(sent_pos)).long(), \
               torch.from_numpy(np.asarray(local_features_pos)).float(), \
               torch.from_numpy(np.asarray(global_features_pos)).float(), \
               torch.from_numpy(np.asarray(video_time_pos)[:1].squeeze()/num_frames_pos).float(), \
               torch.from_numpy(np.asarray(sent_neg)).long(), \
               torch.from_numpy(np.asarray(local_features_neg)).float(), \
               torch.from_numpy(np.asarray(global_features_neg)).float(), \
               torch.from_numpy(np.asarray(video_time_neg)/num_frames_neg).float()

# This should return the full video & the query -> the code should evaluate how to generate the localizations
class EvalDataset(data.Dataset):
  'Characterizes a very Basic Random Video dataset'
  def __init__(self, partition, vocab):
        'Initialization: Video Id == Sentence ID uniquely determines a data point'
        self.annotation_IDs = []
        self.annotation_to_video = {}
        self.annotation_to_times = {}
        self.annotation_to_text = {}
        self.annotation_to_time = {}
        self.vocab = vocab
        self.rgb_file = None
        #self.flow_file = '../data/didemos/average_global_flow.h5'
        self.flow_file = '../data/didemos/average_fc7.h5'
        self.video_to_flow = {}
        self.partition = partition

        with open(self.partition) as data_file:
            data = json.load(data_file)
            for row in data:
                self.annotation_IDs.append(row['annotation_id'])
                self.annotation_to_video[row['annotation_id']] = row['video']
                self.annotation_to_text[row['annotation_id']] = row['description']
                self.annotation_to_time[row['annotation_id']] = row['times'] # this has to be a list now

        with h5py.File(self.flow_file, 'r') as f:
            for key in f.keys():
                data = list(f[key])
                self.video_to_flow[key] = np.asarray(data)

  def __len__(self):
        'Denotes the total number of video-sentence pairs'
        return len(self.annotation_IDs)

  def __getitem__(self, index):
        'Generates one sample of video-sentence pair'
        # Select annotation -> then go the video -> and get the corresponding clips
        # The feature for a video is 6 x 1024 -> this means it is clip-wise, why is it clip wise?
        video_id = self.annotation_to_video[self.annotation_IDs[index]]
        #print(video_id)
        # No matter what the time duration is, you will anyways mean pool things here
        # Based on this mean pool, there is a global feature, there is a local feature andt there are timepoint indices
        # Right now, I can just take send one frame
        # print(video_time)
        # print("Shape of the Video data = {}".format(np.asarray(self.video_to_flow[video_id_pos][int(video_time_pos[0]):int(video_time_pos[1])+1, :]).shape))
        local_features = self.video_to_flow[video_id] # This just returns the first part of the clip -> should mean pool for m$
        #print("Local Features = {}".format(local_features.shape))
        global_features = np.mean(self.video_to_flow[video_id], axis=0)
        #print("Global - Local = {} {}".format(global_features_pos.shape, local_features_pos.shape))
        num_frames = self.video_to_flow[video_id].shape[1]
        # In data loader batch has to be constructed here
        sent = self.vocab.src.numberize(self.annotation_to_text[self.annotation_IDs[index]])
        # Pad/Limit Sentences to a max limit -> see the dataset statistics -> paper says 30?
        max_len = 75
        sent = sent[:max_len]
        sent = sent + [0]*(max_len - len(sent))
        # Stats for Debugging
        #print("Video ID: ", video_id)
        #print("Sentence: ", np.asarray(sent).shape)
        #print("Features: ", np.asarray(features).shape)
        #print("Video Time: ", np.asarray(video_time).shape)

        #print("Feature Sizes = {} {} {} {} {} {} {} {}".format(np.asarray(sent_pos).shape, np.asarray(local_features_pos).shape, np.asarray(global_features_pos).shape, np.asarray(video_time_pos).shape, $
        # return sentence, local feature is the full tensor,
        return torch.from_numpy(np.asarray(sent)).long(), \
               torch.from_numpy(np.asarray(local_features)).float(), \
               torch.from_numpy(np.asarray(global_features)).float(), \
               self.annotation_IDs[index]
