import torchtext
from torchtext.legacy.data import BucketIterator, Field, TabularDataset, Iterator

import spacy
import torch

# set tokens
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_freq = 5

file_word_pad_size = 15
patch_ab_pad_size = 200
patch_content_pad_size = 200
desc_pad_size = 100
title_pad_size = 100
pad_size = file_word_pad_size + patch_ab_pad_size + patch_content_pad_size + desc_pad_size + title_pad_size


print('Start creating dataset!')

FILE_WORD = Field(lower=True, batch_first=True, fix_length=15)
PATCH_AB = Field(lower=True, batch_first=True, fix_length=200)
PATCH_CONTEXT = Field(lower=True, batch_first=True, fix_length=200)
DESC = Field(lower=True, batch_first=True, fix_length=100)
TITLE = Field(lower=True, batch_first=True, fix_length=100)
TRG = Field(tokenize=lambda x: x.split('|'), unk_token='<unk>')


fields = [('review_id', TRG),('patch_word_ab_key_set', PATCH_AB),
         ('patch_word_context_key', PATCH_CONTEXT),
         ('file_word_set', FILE_WORD), ('r_title_cut', TITLE),
         ('r_desc_cut', DESC)]


train_data, valid_data, test_data = TabularDataset.splits(
        path='./data/',
        train='train.csv', 
        validation='eval.csv', 
        test='test.csv', 
        format='csv',
        skip_header=True,
        fields=fields)


train_iter, valid_iter= BucketIterator.splits(
        (train_data, valid_data),
        batch_sizes=(32, 32),
        sort_key=lambda x: len(x.file_word_set),
        sort_within_batch=False,
        device=device,
        shuffle=True)


test_iter = Iterator(
        test_data,    
        batch_size=100, 
        device=device, 
        sort=False,
        shuffle=False,
        sort_within_batch=False, 
        repeat=False)

FILE_WORD.build_vocab(train_data, min_freq=2)
PATCH_AB.build_vocab(train_data, min_freq=2)
PATCH_CONTEXT.build_vocab(train_data, min_freq=2)
DESC.build_vocab(train_data, min_freq=2)
TITLE.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=1)

src_pad_idx = PATCH_CONTEXT.vocab.stoi[PATCH_CONTEXT.pad_token]
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]


file_word_vocab_size = len(FILE_WORD.vocab)
patch_ab_vocab_size = len(PATCH_AB.vocab)
patch_content_vocab_size = len(PATCH_CONTEXT.vocab)
desc_vocab_size = len(DESC.vocab)
title_vocab_size = len(TITLE.vocab)
trg_vocab_size = len(TRG.vocab)

print('file_word_vocab_size len:',file_word_vocab_size)
print('desc_vocab_size len:',desc_vocab_size)
print('title_vocab_size len:',title_vocab_size)
print('patch_content_vocab_size len:',patch_content_vocab_size)
print('patch_ab len:',patch_ab_vocab_size, 'trg len:', trg_vocab_size)
print('Finish creating dataset!')

