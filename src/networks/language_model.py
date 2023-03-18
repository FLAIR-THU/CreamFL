import os
import pickle

import torch
import torch.nn as nn
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

try:
    from pie_model import PIENet
except ImportError:
    try:
        from models.pie_model import PIENet
    except:
        from src.networks.models.pie_model import PIENet


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, wemb_type='glove', word_dim=300, embed_dim=2048, num_class=4, scale=128, mlp_local=False):
        super(EncoderText, self).__init__()
        with open('src/datasets/vocabs/coco_vocab.pkl',
                  'rb') as fin:
            vocab = pickle.load(fin)
        word2idx = vocab['word2idx']

        self.embed_dim = embed_dim

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)
        if torch.cuda.is_available():
            self.pie_net = self.pie_net.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.class_fc = nn.Linear(embed_dim, num_class)
        self.class_fc_2 = nn.Linear(embed_dim, 80)

        self.init_weights(wemb_type, word2idx, word_dim)
        self.is_train = True
        self.phase = ''
        self.scale = scale

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
            )

    def init_weights(self, wemb_type, word2idx, word_dim, cache_dir=os.environ['HOME'] + '/data/'):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim, f'wemb.vectors.shape {wemb.vectors.shape}'

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        # Embed word ids to vectors
        wemb_out = self.embed(x)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I.to(x.device)).squeeze(1)

        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        # print('1', out.device, wemb_out.device, pad_mask.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask.to(out.device))
        out = out * self.scale
        out = self.relu(out)

        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc.weight)
            self.class_fc.weight.data = fc_weight_relu
            x = self.class_fc(out)

            fc_weight_relu2 = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu2
            x2 = self.class_fc_2(out)

            return x, x2, fc_weight_relu, fc_weight_relu2

        if self.mlp_local:
            out = self.head_proj(out)

        out = F.normalize(out, p=2, dim=1)
        return out


if __name__ == '__main__':
    models_name = ['LSTM']

    input = torch.zeros((32, 30)).long().cuda()
    lengths = (torch.ones(32) * 29).long().cuda()

    for m in models_name:
        print('running', m)
        model = LModel(m, cls_num=10).cuda()

        print(model(input, lengths).shape)
    print('runningEncoderText')

    model = EncoderText().cuda()
    # print(model(input, lengths)['embedding'].shape)
    print(model(input, lengths).shape)
