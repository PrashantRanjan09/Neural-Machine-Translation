from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
from Encoder import EncoderRNN
from Decoder import DecoderRNN
from AttnDecoder import AttnDecoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from helper_functions import prepareData,readLangs,filterPairs,trainIters,evaluate

use_cuda = torch.cuda.is_available()
'''
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
'''
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

teacher_forcing_ratio = 0.5

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(input_lang,output_lang,encoder1, attn_decoder1,pairs, n_iters=75000, print_every=5000)

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
#evaluateAndShowAttention ("Your input sentence to be translated")
