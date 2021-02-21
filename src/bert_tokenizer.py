import pandas as pd
import transformers
from transformers import BertTokenizer, AutoTokenizer
import json
from tqdm import tqdm

def get_impressions_from_csv(path):	
        df = pd.read_csv(path)
        imp = df['Report Impression']
        imp = imp.str.strip()
        imp = imp.replace('\n',' ', regex=True)
        imp = imp.replace('\s+', ' ', regex=True)
        imp = imp.str.strip()
        return imp

def tokenize(impressions, tokenizer):
        new_impressions = []
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
        for i in tqdm(range(impressions.shape[0])):
                tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
                if tokenized_imp: #not an empty report
                        res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                        if len(res) > 512: #length exceeds maximum size
                                #print("report length bigger than 512")
                                res = res[:511] + [tokenizer.sep_token_id]
                        new_impressions.append(res)
                else: #an empty report
                        new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
        return new_impressions

def load_list(path):
        with open(path, 'r') as filehandle:
                impressions = json.load(filehandle)
                return impressions

if __name__ == "__main__":
        tokenizer = BertTokenizer.from_pretrained('/data3/aihc-winter20-chexbert/bluebert/pretrain_repo')
        #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        #tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

        impressions = get_impressions_from_csv('/data3/aihc-winter20-chexbert/chexpert_data/vision_test_gt.csv')
        new_impressions = tokenize(impressions, tokenizer)
        with open('/data3/aihc-winter20-chexbert/bluebert/vision_labels/impressions_lists/vision_test', 'w') as filehandle:
                json.dump(new_impressions, filehandle)
