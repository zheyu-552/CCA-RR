import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import sys
import numpy as np
import argparse
import configs
from loguru import logger
from tqdm import tqdm
from datetime import datetime
from datasets import train_iter, valid_iter, test_iter
from datasets import src_pad_idx, trg_pad_idx
from datasets import file_word_vocab_size, patch_ab_vocab_size, patch_content_vocab_size ,desc_vocab_size, title_vocab_size, trg_vocab_size
from datasets import file_word_pad_size, patch_ab_pad_size, patch_content_pad_size, desc_pad_size, title_pad_size, pad_size
import method.jointemb as jointemb
from utils import validate, evaluation, validate1, multi_label_metrics_transfer, binary_acc, mrr_metrix
import time

#logger.add("./data/train.log")


def train_epoch(model, train_data, optimizer, config, args, criteon, device):
    """Epoch operation in the training pharse."""

    model.train()
    losses = []
    acces = []
    acces_1, acces_3, acces_5, acces_10 = [], [], [], []
    mrres_10 = []
    itr_start_time = time.time()
    n_itr = len(train_data)
    text = "---------- Training ----------"
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    for batch in tqdm(train_data, mininterval=2, desc=text, leave=False):
    #for index, batch in enumerate(train_data):
        review_id = batch.review_id
        patch_ab = batch.patch_word_ab_key_set
        file_word = batch.file_word_set
        patch_content = batch.patch_word_context_key
        desc = batch.r_desc_cut
        title = batch.r_title_cut
        
#         print('file_word and review_id:', file_word.shape, review_id.shape, pad_size)
        out = model(patch_ab, file_word, patch_content, desc, title)
        review_id_t = multi_label_metrics_transfer(review_id, trg_vocab_size, device)
#         print('='*30)
#         print(out)
#         out = out[:,2:] # torch.cat((out[:,0].reshape(-1,1), out[:,2:]), 1)
#         print(out.shape, review_id_t.shape)
#         print(review_id)
#         print('t:',review_id_t)
        review_id_t[:, 1] = 0
#         print(review_id_t)
        loss = criteon(out, review_id_t.float())
        acc = binary_acc(out, review_id, 1)
        
        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        acces.append(acc)
        
#         acces_1.append(binary_acc(out, review_id, 1))
#         acces_3.append(binary_acc(out, review_id, 3))
#         acces_5.append(binary_acc(out, review_id, 5))
#         acces_10.append(binary_acc(out, review_id, 10))
#         mrres_10.append(mrr_metrix(out, review_id))
        
        losses.append(loss.item())

    return losses, acces #acces_1, acces_3, acces_5, acces_10, mrres_10 # acces


def eval_epoch(model, valid_data, criteon, device):
    """Valid model when finishing a epoch in the training phrase."""
    losses = []
    acces_1, acces_3, acces_5, acces_10 = [], [], [], []
    mrres_10 = []
    text = '---------- Validation ----------'
    with torch.no_grad():
        for batch in tqdm(valid_data, mininterval=2, desc=text, leave=False):
            
            review_id = batch.review_id
            patch_ab = batch.patch_word_ab_key_set
            file_word = batch.file_word_set
            patch_content = batch.patch_word_context_key
            desc = batch.r_desc_cut
            title = batch.r_title_cut
            out = model(patch_ab, file_word, patch_content, desc, title)
            
            review_id_t = multi_label_metrics_transfer(review_id, trg_vocab_size, device)
#             out = out[:,2:]#torch.cat((out[:,0].reshape(-1,1), out[:,2:]), 1)
#         print('out and review_id:', out.shape, review_id_t.shape)
#             print(review_id)
#         print('t:',review_id_t)
            review_id_t[:, 1] = 0
#             print(review_id_t)
            loss = criteon(out, review_id_t.float())
#             binary_acc(out, review_id, 1) # , True
            acces_1.append(binary_acc(out, review_id, 1))
            acces_3.append(binary_acc(out, review_id, 3))
            acces_5.append(binary_acc(out, review_id, 5))
            acces_10.append(binary_acc(out, review_id, 10))
            mrres_10.append(mrr_metrix(out, review_id))
            losses.append(loss.item())
    
    return losses, acces_1, acces_3, acces_5, acces_10, mrres_10


def train(model, training_data, validation_data, optimizer, args, criteon, device):
    """Training."""
    logger.info("Start training!")
    best_loss = 1e10
    config = getattr(configs, 'config')()

    def save_model(model, path):
        torch.save(model.state_dict(), path)

    for epoch in range(args.Epoch):
        info = '[ Epoch ' + str(epoch) + ' ]'
        logger.info(info)
#         train_loss, train_acc_1, train_acc_3, train_acc_5, train_acc_10, train_mrr_10 = train_epoch(model, training_data, optimizer, config, args, criteon, device) # train_acc
        train_loss, train_acc = train_epoch(model, training_data, optimizer, config, args, criteon, device) # train_acc
        logger.info("The loss of epoch {} is: {}".format(epoch, np.mean(train_loss)))
#         logger.info("The acc_1_3_5_10 MRR_10 of epoch {} is: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}".format(epoch, np.mean(train_acc_1), np.mean(train_acc_3), np.mean(train_acc_5), np.mean(train_acc_10),np.mean(train_mrr_10)))
        logger.info("The acc_10 of epoch {} is: {}".format(epoch, np.mean(train_acc)))

        logger.info("Validating.")
        loss_per_batch, valid_acc_1, valid_acc_3, valid_acc_5, valid_acc_10, valid_mrr_10 = eval_epoch(model, validation_data, criteon, device)
        logger.info("Validation loss is: {}".format(np.mean(loss_per_batch)))
        logger.info("The acc_1_3_5_10 MRR_10 of epoch {} is: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}".format(epoch, np.mean(valid_acc_1), np.mean(valid_acc_3), np.mean(valid_acc_5), np.mean(valid_acc_10),np.mean(valid_mrr_10)))
        
#         t_results = "train_epoch: {}, loss = {}, top1 = {}, top3 = {}, top5 = {}, top10 = {}, mrr = {}".format(epoch, np.mean(train_loss), np.mean(train_acc_1), np.mean(train_acc_3), np.mean(train_acc_5), np.mean(train_acc_10),np.mean(train_mrr_10))
        v_results = "epoch: {}, loss = {}, top1 = {}, top3 = {}, top5 = {}, top10 = {}, mrr = {}".format(epoch, np.mean(loss_per_batch), np.mean(valid_acc_1), np.mean(valid_acc_3), np.mean(valid_acc_5), np.mean(valid_acc_10), np.mean(valid_mrr_10))
        with open('result_android_bert.txt', 'a') as f:
            f.write(v_results + '\n') # t_results + '\n' + 

#         valid_loss = loss_per_batch
#         if best_loss > valid_loss:
#             best_loss = valid_loss
#             logger.info("The current best loss is: {}".format(best_loss))
#             path = args.model_path + 'joint_embed_model_128_dc.h5'
#             save_model(model, path)

#         if (epoch+1) % 5  == 0:
#             optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate']*0.1, eps=config['adam_epsilon'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='JointEmbedder', help='model name')
    parser.add_argument('--dataset', type=str, default='dataset/', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('--model_path', type=str, default='./model_save/', help='path of saving model')
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--best_mrr', type=float, default=0., help='The MRR metric.')

    parser.add_argument('--log_every', type=int, default=1000, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=300000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')

    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')
    parser.add_argument('--Epoch', type=int, default=300, help="Training Epoch")

    args = parser.parse_args()

    config = getattr(configs, 'config')()
#     config['src_vocab'] = src_vocab_size
    config['file_word_vocab'] =  file_word_vocab_size
    config['patch_ab_vocab'] = patch_ab_vocab_size
    config['patch_content_vocab'] = patch_content_vocab_size
    config['desc_vocab'] = desc_vocab_size
    config['title_vocab'] = title_vocab_size
# 保持不变
    config['trg_vocab'] = trg_vocab_size
    config['src_pad_idx'] = src_pad_idx
    config['trg_pad_idx'] = trg_pad_idx
    config['pad_size'] = pad_size
#     file_word_pad_size patch_ab_pad_size patch_content_pad_size desc_pad_size title_pad_size
    config['file_word_pad_size'] = file_word_pad_size
    config['patch_ab_pad_size'] = patch_ab_pad_size
    config['patch_content_pad_size'] = patch_content_pad_size
    config['desc_pad_size'] = desc_pad_size
    config['title_pad_size'] = title_pad_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(jointemb, args.model)(config)
    criteon = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=config['adam_epsilon'])

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))

    def count_parameters(model):
        return sum((p.numel() for p in model.parameters() if p.requires_grad))

    print(f'The model has {count_parameters(model):,} trainable parameters!')

    if args.mode == 'train':
        model = model.to(device)
        train(model, train_iter, valid_iter, optimizer, args, criteon, device)
    elif args.mode == 'eval':
        path = args.model_path + 'joint_embed_model_128_dc.h5'
        load_model(model, path, device)
        model.to(device)
        K = [10]
        for k in K:
            start_time = time.time()
            re, acc, mrr, map, ndcg = validate1(test_iter, model, k, 'cos', config, device)
            #MMR = evaluation(test_iter, model, mode='cosine')
            search_time = time.time() - start_time
            query_time = search_time / config['test_data_leng']
            results = "re = {}, mrr = {}, ndcg = {}".format(re, mrr, ndcg)
            with open('result_joint_128_dc.txt', 'a') as f:
                f.write(results + '\n')
            print("The search time of each query is: {}".format(query_time))


if __name__ == "__main__":
    main()
