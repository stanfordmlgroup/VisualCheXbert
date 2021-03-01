import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from models.bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from datasets.unlabeled_dataset import UnlabeledDataset
from constants import *
from tqdm import tqdm
import pickle

def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def apply_logreg_mapping(df_probs, logreg_models_path):
    logreg_models = {}
    visualchexbert_dict = {}
    try:
        with open(logreg_models_path, "rb") as handle:
            logreg_models = pickle.load(handle)
    except Exception as e:
        print("Error loading path to logistic regression models. Please ensure that the pickle file is in the checkpoint folder.")
        print(f"Exception: {e}")
    for condition in CONDITIONS:
        clf = logreg_models[condition]
        y_pred = clf.predict(df_probs)
        visualchexbert_dict[condition] = y_pred
    df_visualchexbert = pd.DataFrame.from_dict(visualchexbert_dict)
    return df_visualchexbert

def label_and_save_preds(checkpoint_folder, csv_path, out_path):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports
    @param out_path (string): path to output directory

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path)
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{checkpoint_folder}/visualCheXbert.pth"
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = torch.sigmoid(out[j]) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
             
    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)

    # Apply mapping from probs to image labels
    logreg_models_path = f"{checkpoint_folder}/logreg_models.pickle"
    df_visualchexbert = apply_logreg_mapping(df, logreg_models_path)

    reports = pd.read_csv(csv_path)['Report Impression']

    df_visualchexbert['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df_visualchexbert = df_visualchexbert[new_cols]
    df_visualchexbert.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)

    return df_visualchexbert

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint_folder', type=str, nargs='?', required=False, default="checkpoint",
                        help='path to folder with pytorch model checkpoints and serialized log reg models')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint_folder

    label_and_save_preds(checkpoint_path, csv_path, out_path)
