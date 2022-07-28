from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Encoder import Encoder
from attention.JointEncoder import JointEncoder
from utils import get_pad_mask


class JointEmbedder(nn.Module, ABC):
    """
    Joint embedding for code snippets and code description, and have a extra
    attention to code snippet components to build internal relationship.
    """

    def __init__(self, config):
        super(JointEmbedder, self).__init__()

        self.conf = config
        self.trg_len = config['trg_vocab']
        self.margin = config["margin"]
        
        self.file_word_enc = Encoder(config['file_word_vocab'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['src_pad_idx'])
        
        self.patch_ab_enc = Encoder(config['patch_ab_vocab'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['src_pad_idx'])
        
        self.patch_content_enc = Encoder(config['patch_content_vocab'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['src_pad_idx'])
        
        self.desc_enc = Encoder(config['desc_vocab'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['src_pad_idx'])
        
        self.title_enc = Encoder(config['title_vocab'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['src_pad_idx'])

        self.fc = nn.Linear(config["d_model"] * config["pad_size"], config['trg_vocab'])
    
#         file_word_pad_size patch_ab_pad_size patch_content_pad_size desc_pad_size title_pad_size
        self.patch_ab_fc = nn.Linear(config["d_model"] * config["patch_ab_pad_size"], config['d_model'])
        self.file_word_fc = nn.Linear(config["d_model"] * config["file_word_pad_size"], config['d_model'])
        self.patch_content_fc = nn.Linear(config["d_model"] * config["patch_content_pad_size"], config['d_model'])
        self.desc_fc = nn.Linear(config["d_model"] * config["desc_pad_size"], config['d_model'])
        self.title_fc = nn.Linear(config["d_model"] * config["title_pad_size"], config['d_model'])
        
        self.joint_fc1 = nn.Linear(config["d_model"] * 5, config["d_model"] * 2) # config["pad_size"]
        
        self.joint_fc2 = nn.Linear(config["d_model"] * 2, config["d_model"] *2)
        
        self.joint_fc3 = nn.Linear(config["d_model"] * 2, config['trg_vocab'])

        self.dropout = nn.Dropout(p=0.3) #0.3
        self.ReLU = nn.ReLU(inplace=True)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def file_word_encoding(self, file_word):
        
        file_word_repr, file_word_mask = self.file_word_enc(file_word)

        return file_word_repr, file_word_mask
    
    def patch_ab_encoding(self, patch_ab):
        
        patch_ab_repr, patch_ab_mask = self.patch_ab_enc(patch_ab)

        return patch_ab_repr, patch_ab_mask
    
    def patch_content_encoding(self, patch_content):
        
        patch_content_repr, patch_content_mask = self.patch_content_enc(patch_content)

        return patch_content_repr, patch_content_mask
    
    def desc_encoding(self, desc):
        
        desc_repr, desc_mask = self.desc_enc(desc)

        return desc_repr, desc_mask
    
    def title_encoding(self, title):
        
        title_repr, title_mask = self.title_enc(title)

        return title_repr, title_mask
        
    
    def criterion_loss(self, ):
        criteon = nn.BCEWithLogitsLoss()

    
    def forward(self, patch_ab, file_word, patch_content, desc, title):

        file_word_repr, _ = self.file_word_encoding(file_word) # code_repr, code_mask
        patch_ab_repr, _ = self.patch_ab_encoding(patch_ab)
        patch_content_repr, _ = self.patch_content_encoding(patch_content)
        desc_repr, _ = self.desc_encoding(desc)
        title_repr, _ = self.title_encoding(title)

        file_word_repr = file_word_repr.view(file_word_repr.size(0), -1)
        patch_ab_repr = patch_ab_repr.view(patch_ab_repr.size(0), -1)
        patch_content_repr = patch_content_repr.view(patch_content_repr.size(0), -1)
        desc_repr = desc_repr.view(desc_repr.size(0), -1)
        title_repr = title_repr.view(title_repr.size(0), -1)

        file_word_repr = self.file_word_fc(file_word_repr)
        patch_ab_repr = self.patch_ab_fc(patch_ab_repr)
        patch_content_repr = self.patch_content_fc(patch_content_repr)
        desc_repr = self.desc_fc(desc_repr)
        title_repr = self.title_fc(title_repr)
        
        joint_repr = torch.cat((file_word_repr, patch_ab_repr, patch_content_repr, desc_repr, title_repr), 1)

        joint_repr = self.dropout(joint_repr) # joint_repr
        joint_repr = self.joint_fc1(joint_repr)
        joint_repr = self.ReLU(joint_repr)
        joint_repr = self.dropout(joint_repr)
        joint_repr = self.joint_fc2(joint_repr)
        joint_repr = self.ReLU(joint_repr)
        out = self.joint_fc3(joint_repr)

        
        return out

