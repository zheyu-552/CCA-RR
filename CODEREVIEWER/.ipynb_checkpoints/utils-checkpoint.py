from abc import ABC

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
from scipy.spatial.distance import cdist


class PositionlEncoding(nn.Module, ABC):

    def __init__(self, d_hid, n_position=500):
        super(PositionlEncoding, self).__init__()

        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def normalize(data):
    """normalize matrix by rows"""
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def validate1(test_iter, model, k, sim_measure, config, device):
    """
    To evaluate the performance of our trained model.
    :param valid_set: Use it to evaluate trained model.
    :param model: Our trained model.
    :param k: Take the top-k results.
    :param sim_measure: The way to calculate similarity, in our experiments,
                        we use cosine similarity.
    :param device: Whether using GPU.
    :return:
    """

    def recall(gold, prediction, results):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index <= results:
                sum += 1
        return sum / float(len(gold))

    def acc(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum += 1
        return sum / float(len(gold))

    def map(gold, prediction):
        sum = 0.
        for idx, val in enumerate(gold):
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + (idx + 1) / float(index + 1)
        return sum / float(len(gold))

    def mrr(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
        return sum / float(len(gold))

    def ndcg(gold, prediction):
        dcg = 0.
        idcgs = idcg(len(gold))
        for i, predictItem in enumerate(prediction):
            if predictItem in real:
                item_relevance = 1
                rank = i + 1
                dcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcgs)

    def idcg(n):
        idcg = 0
        item_relevance = 1
        for i in range(n):
            idcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    model.eval()
    
    with open('./cos/group_list.txt') as f:
        group_list = f.readlines()
    re, accu, mrrs, maps, ndcgs = 0., 0., 0., 0., 0.
#     code_total = torch.zeros((config['test_data_leng'], config['d_model']))
#     desc_total = torch.zeros((config['test_data_leng'], config['d_model']))
    b_s = config['test_batch_size']
    data_len = config['test_data_leng']
    for idx, batch in tqdm(enumerate(test_iter)):
       # batch includes code, pos_desc, and neg_desc.
        with torch.no_grad():
            code = batch.desc.to(device) #.transpose(0, 1)
            desc = batch.file.to(device) #.transpose(0, 1)
            code_repr, code_mask = model.code_encoding(code)  # [batch_size, feature_dim]
            desc_repr, desc_mask = model.desc_encoding(desc)  # [batch_size, feature_dim]
#             code_repr = torch.mean(code_repr, dim=1)
#             desc_repr = torch.mean(desc_repr, dim=1)
            code_repr, desc_repr = model.joint_encoding(code_repr, desc_repr, code_mask, desc_mask)
            code_repr = code_repr.data #.cpu()
            desc_repr = desc_repr.data #.cpu()
#             desc_vec = desc_total[i].unsqueeze(0)  # [1 x dim]
            n_results = k
            if sim_measure == 'cos':
                sims = torch.cosine_similarity(code_repr, desc_repr, dim=1)  # [data_len, 1]
            _, predict_origin = sims.topk(k=b_s)
            predict = predict_origin[:n_results]
            predict = [int(k) for k in predict]
            predict_origin = [int(k) for k in predict_origin]
            real = list(range(len(group_list[idx].split())))
#             results.append(predict[0])
            re += recall(real, predict_origin, n_results)
            accu += acc(real, predict)
            mrrs += mrr(real, predict)
            maps += map(real, predict)
            ndcgs += ndcg(real, predict)
            
    re = re / float(data_len) #float(b_s)
    accu = accu / float(len(test_iter)) # float(b_s)
    mrrs = mrrs / float(len(test_iter)) # float(b_s)
    maps = maps / float(len(test_iter)) # loat(b_s)
    ndcgs = ndcgs / float(len(test_iter))  #float(b_s)
    print(re, accu, mrrs, maps, ndcgs)
    return re, accu, mrrs, maps, ndcgs


def validate(test_iter, model, k, sim_measure, config, device):
    """
    To evaluate the performance of our trained model.
    :param valid_set: Use it to evaluate trained model.
    :param model: Our trained model.
    :param k: Take the top-k results.
    :param sim_measure: The way to calculate similarity, in our experiments,
                        we use cosine similarity.
    :param device: Whether using GPU.
    :return:
    """

    def recall(gold, prediction, results):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index <= results:
                sum += 1
        return sum / float(len(gold))

    def acc(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum += 1
        return sum / float(len(gold))

    def map(gold, prediction):
        sum = 0.
        for idx, val in enumerate(gold):
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + (idx + 1) / float(index + 1)
        return sum / float(len(gold))

    def mrr(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
        return sum / float(len(gold))

    def ndcg(gold, prediction):
        dcg = 0.
        idcgs = idcg(len(gold))
        for i, predictItem in enumerate(prediction):
            if predictItem in real:
                item_relevance = 1
                rank = i + 1
                dcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcgs)

    def idcg(n):
        idcg = 0
        item_relevance = 1
        for i in range(n):
            idcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    model.eval()
    re, accu, mrrs, maps, ndcgs = 0., 0., 0., 0., 0.
    code_total = torch.zeros((config['test_data_leng'], config['d_model']))
    desc_total = torch.zeros((config['test_data_leng'], config['d_model']))
    b_s = config['batch_size']
    data_len = config['test_data_leng']
    for idx, batch in enumerate(test_iter):
       # batch includes code, pos_desc, and neg_desc.
        with torch.no_grad():
            code_repr, code_mask = model.code_encoding(batch.desc)  # [batch_size, feature_dim]
            desc_repr, desc_mask = model.desc_encoding(batch.file)  # [batch_size, feature_dim]
            code_repr, desc_repr = model.joint_encoding(code_repr, desc_repr, code_mask, desc_mask)
            code_repr = code_repr.data.cpu()
            desc_repr = desc_repr.data.cpu()
            if code_repr.size(0) == b_s:
                code_total[idx*b_s:(idx+1)*b_s, :] = code_repr
                desc_total[idx*b_s:(idx+1)*b_s, :] = desc_repr
            else:
                print("low batch!")
                code_total[idx*b_s:, :] = code_repr
                desc_total[idx*b_s:, :] = desc_repr
    print("Finishing embedding for code and desc")
    code_total = code_total.cuda()
    desc_total = desc_total.cuda()
    results = []
    for i in tqdm(range(data_len)):  # for i in range(pool_size):
        desc_vec = desc_total[i].unsqueeze(0)  # [1 x dim]
        n_results = k
        if sim_measure == 'cos':
            sims = torch.cosine_similarity(code_total, desc_vec, dim=1)  # [data_len, 1]

        #neg_sims = np.negative(sims)
        #predict_origin = np.argsort(neg_sims)
        #predict = np.argpartition(negsims, kth=n_results - 1)
        _, predict_origin = sims.topk(k=data_len)
        predict = predict_origin[:n_results]
        predict = [int(k) for k in predict]
        predict_origin = [int(k) for k in predict_origin]
        real = [i]
        results.append(predict[0])
        re += recall(real, predict_origin, n_results)
        accu += acc(real, predict)
        mrrs += mrr(real, predict)
        maps += map(real, predict)
        ndcgs += ndcg(real, predict)
#         if (i+1) % 100 == 0:
#             result = "The score of MRR is {}".format(mrrs / float(i+1))
#             print(result)
    with open('results', 'a') as f:
        for each in results:
            f.write(str(each) + '\n')
    re = re / float(data_len)
    accu = accu / float(data_len)
    mrrs = mrrs / float(data_len)
    maps = maps / float(data_len)
    ndcgs = ndcgs / float(data_len)
    return re, accu, mrrs, maps, ndcgs


def evaluation(test_iter, model, mode):
    
    code_total = []
    desc_total = []
    b_s = 1000
    
    def compute_ranks(code_repr, desc_repr, metric):
        
        distances = cdist(code_repr, desc_repr, metric=metric)
        
        correct_elements = np.expand_dims(np.diag(distances), axis=-1)
        
        return np.sum(distances <= correct_elements, axis=-1), distances
    
    for batch in test_iter:
       # batch includes code, pos_desc, and neg_desc.
        with torch.no_grad():
            code = batch.desc
            desc = batch.file
            idxs = torch.randperm(code.size(0))
            code = code[idxs]
            desc = desc[idxs]
            code_repr = model.code_encoding(code)  # [batch_size, feature_dim]
            desc_repr = model.desc_encoding(desc)  # [batch_size, feature_dim]
            code_repr, desc_repr, _ = model.joint_encoding(code_repr, desc_repr, desc_repr)
            code_repr = code_repr.data.cpu()
            desc_repr = desc_repr.data.cpu()
            if code_repr.size(0) == b_s:
                code_total.append(code_repr)
                desc_total.append(desc_repr)
            if code_repr.size(0) != b_s:
                break
    
    mmr_total = 0.
    
    for i in range(len(code_total)):
        
        ranks, disctances = compute_ranks(code_total[i], desc_total[i], mode)
        
        mmr_total += np.mean(1. / ranks)
    
    MMR = mmr_total / (len(code_total))
    
    print("The MMR score is : {}".format(MMR))
    
    return MMR

def multi_label_metrics_transfer(y, label_num, device):
    re_tensor = torch.zeros(
    y.shape[1],
    label_num,
    dtype=torch.int64,
    device=device,
    ).scatter_(
        1,
        y.T,
        torch.ones(
            y.shape[1],
            label_num,
            dtype=torch.int64,
            device=device,
         ),
    )
    
    return re_tensor

def binary_acc(preds, y, k, state=False):
    acc_10 = []
    _,maxk = torch.topk(preds, k, dim=-1)

    y_t = y.T
    for col in range(len(maxk)):
        acc_10.append(acc(y_t[col], maxk[col].tolist(), state))
    mean_acc = np.array(acc_10).mean()
#     print(mean_acc)
    return mean_acc

def acc(gold, prediction, state):
    gold = list(filter(lambda num: num !=  1, gold))
#     gold = [i-2 for i in gold]
    if state:
        print(gold, prediction)
    for idx, val in enumerate(gold):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = 1
                break
    return sum

def mrr_metrix(preds, y):
    mrr_k = []

    _,maxk = torch.topk(preds, len(preds), dim=-1) 
    y_t = y.T
    for col in range(len(preds)):
        mrr_k.append(mrr(y_t[col], maxk[col].tolist()))
        
    mean_mrr = np.array(mrr_k).mean()
#     print(mean_acc)
    return mean_mrr


def mrr(gold, prediction):
    sum = 0
    gold = list(filter(lambda num: num !=  1, gold))
    for val in gold:
        try:
            index = prediction.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)

    return sum 
