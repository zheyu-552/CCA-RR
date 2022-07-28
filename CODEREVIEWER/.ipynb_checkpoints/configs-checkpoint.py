#!/usr/bin/env python
# encoding: utf-8

def config():

    conf = {
        # data info
        'test_data_leng': 59100,
        'test_batch_size':100,
        # vocabulary info
        'file_word_vocab': None,
        'patch_ab_vocab': None,
        'patch_content_vocab': None,
        'desc_vocab': None,
        'title_vocab': None,
#         'src_vocab': None,
        'trg_vocab': None,
        'src_pad_idx': None,
        'trg_pad_idx': None,

        # training parameters
        'batch_size': 64,
        'Epoch': 100,
        'learning_rate': 1e-3, # 5e-8 1e-7
        'adam_epsilon': 1e-8, # 5e-8
        'warmup_steps': 5000,

        # model parameters
        'd_word_dim': 256, # 256
        'd_model': 256, # 256
        'd_ffn': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_k': 32,
        'd_v': 32, # 32 32 256
        'margin': 2.17,
        'sim_measure': 'softmax'
    }
    return conf
