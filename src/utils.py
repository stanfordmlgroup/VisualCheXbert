import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from models.bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from constants import *

def generate_attention_masks(batch, source_lengths, device):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)

def compute_auc(y_true, y_pred):
    """Compute the positive auc score
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions 

    @returns res (list): List of 14 scalars
    """
    res = []
    for j in range(len(y_true)):
        if len(set(y_true[j].tolist())) == 1: #only one class present
            res.append(1.0)
        else:
            res.append(roc_auc_score(y_true[j], y_pred[j]))
    return res

def evaluate(model, dev_loader, device, return_pred=False):
    """ Function to evaluate the current model weights
    @param model (nn.Module): the labeler module 
    @param dev_loader (torch.utils.data.DataLoader): dataloader for dev set  
    @param device (torch.device): device on which data should be
    @param return_pred (bool): whether to return predictions or not

    @returns res_dict (dictionary): dictionary with key 'auc' and with value 
                            being a list of length 14 with each element in the 
                            list as a scalar. If return_pred is true then a 
                            tuple is returned with the aforementioned dictionary 
                            as the first item, a list of predictions as the 
                            second item, and a list of ground truth as the 
                            third item
    """
    
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_true = [[] for _ in range(len(CONDITIONS))]
    
    with torch.no_grad():
        for i, data in enumerate(dev_loader, 0):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            label = data['label'] #(batch_size, 14)
            label = label.permute(1, 0).to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)
            
            for j in range(len(out)):
                out[j] = out[j].to('cpu') #move to cpu for sklearn
                curr_y_pred = torch.sigmoid(out[j]) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)
                y_true[j].append(label[j].to('cpu'))
                
            if (i+1) % 200 == 0:
                print('Evaluation batch no: ', i+1)

    for j in range(len(y_true)):
        y_true[j] = torch.cat(y_true[j], dim=0)
        y_pred[j] = torch.cat(y_pred[j], dim=0)

    if was_training:
        model.train()

    auc = compute_auc(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    res_dict = {'auc': auc}
    
    if return_pred:
        return res_dict, y_pred, y_true
    else:
        return res_dict

def test(model, checkpoint_path, test_ld):
    """Evaluate model on test set. 
    @param model (nn.Module): labeler module
    @param checkpoint_path (string): location of saved model checkpoint
    @param test_ld (dataloader): dataloader for test set
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Doing evaluation on test set\n")
    metrics = evaluate(model, test_ld, device)
    auc = metrics['auc']

    print()
    metric_avg = []
    for j in range(len(CONDITIONS)):
        print('%s auc: %.3f' % (CONDITIONS[j], auc[j]))
        if CONDITIONS[j] in EVAL_CONDITIONS:
            metric_avg.append(auc[j])
    print('average of auc: %.3f' % (np.mean(metric_avg)))
    

def label_report_list(checkpoint_path, report_list):
    """ Evaluate model on list of reports.
    @param checkpoint_path (string): location of saved model checkpoint
    @param report_list (list): list of report impressions (string)
    """
    imp = pd.Series(report_list)
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('[0-9]\.', '', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_pred = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_imps = tokenize(imp, tokenizer)
    with torch.no_grad():
        for imp in new_imps:
            # run forward prop
            imp = torch.LongTensor(imp)
            source = imp.view(1, len(imp))
            
            attention = torch.ones(len(imp))
            attention = attention.view(1, len(imp))
            out = model(source.to(device), attention.to(device))

            # get predictions
            result = {}
            for j in range(len(out)):
                curr_y_pred = out[j] #shape is (1)
                result[CONDITIONS[j]] = CLASS_MAPPING[curr_y_pred.item()]
            y_pred.append(result)
    return y_pred

