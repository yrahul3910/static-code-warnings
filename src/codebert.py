# thanks to Zhou Xin for the help and testing out other configurations of CodeBERT
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
from transformers import RobertaTokenizerFast, RobertaModel
import torch
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

import random

# e.g. run as `python codebert.py -project derby`

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
setup_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, choices=['ant', 'cassandra', 'commons', 'derby', 'jmeter', 'lucene-solr', 'tomcat'])
parser.add_argument('-model_name', type=str, default='forest', choices=['tree', 'forest', 'linear'], help='the classifier used; tree -> Decision tree; foresr--> random forest; linear --> Logistic Regression')
parser.add_argument('-only_SE', action='store_true',help='choose to not include method content and do not use CodeBERT embeddings for method content')
parser.add_argument('-only_bert', action='store_true')
parser.add_argument('-codebert_pooling', type=str, default='first', choices=['mean', 'first', 'last'], help='choose the way to aggregate CodeBERT token embeddings into the method embedding')
parser.add_argument('-scaler', type=str, default="standard", choices=['none','standard', 'minmax', 'maxabs', 'quantile'], help='different scaling strategies')  
parser.add_argument('-balance', type=str, default='none',choices=['none','smote', 'downsample', 'upsample'], help= 'how to balance the positives/negatives')
args = parser.parse_args()

folder = './cache'
if not os.path.exists(folder):
    os.makedirs(folder)
folder = './log'
if not os.path.exists(folder):
    os.makedirs(folder)

 
## read single project
train_df = pd.read_csv ('../data/codebert_train/'+ args.project+'_B_features.csv')
test_df = pd.read_csv ('../data/codebert_test/'+ args.project+'_C_features.csv')
train_samples = len(train_df)
test_samples = len(test_df)
all_df = pd.concat([train_df,test_df], axis=0, ignore_index=True)
#print(all_df)
all_df = all_df.dropna(axis=1,how='any')
all_df = all_df
#print(all_df)

def encode_non_numrical_features (df, col_name):
	data_copy = df.copy()
	le = LabelEncoder()
	#print(data_copy[col_name])
	#print(data_copy[col_name])
	le.fit(data_copy[col_name])
	labels = le.transform(data_copy[col_name])
	#print(labels)
	df[col_name] = pd.Series(labels)
	#print(df)
## encoding non-numrical-features as numricals
non_numrical_names = ['url', 'F54', 'F55', 'canonical_id','method_name', 'F26', 'F21', 'F20']
for name_ in non_numrical_names:
	if name_ not in all_df.columns:
	    continue
	encode_non_numrical_features (all_df, name_)

#print(all_df)


## CodeBERT embedding for "method content"
method_contents = all_df["method_content"]
all_df = all_df.drop(['method_content'], axis=1)
possible_codebert_emb_file = './cache/'+args.project+'_all.pkl'

if os.path.exists(possible_codebert_emb_file):
    mean_method_emb, first_method_emb, last_method_emb = pickle.load(open(possible_codebert_emb_file,'rb'))
    print('have codebert embedding in cache')
else:
    print('no codebert embedding in cache')
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    
    def encode_codebert_embedding(input_tokens):
        #print(input_tokens)
        ## Tokenize
        code_tokens=tokenizer.tokenize(input_tokens)[:510]
        #print(len(code_tokens))
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        attention_mask_tokens = [1]*len(source_tokens)
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        outputs = model(input_ids = torch.LongTensor(source_ids).unsqueeze(0), attention_mask=torch.LongTensor(attention_mask_tokens).unsqueeze(0))
        last_hidden_states = outputs.last_hidden_state
        #print('last_hidden_states',last_hidden_states, last_hidden_states.size())

        ## Extract Features
        first_rep = last_hidden_states[0,-1,:]
        last_rep = last_hidden_states[0,0,:]
        mean_rep = torch.mean(last_hidden_states[0], axis=0)
        #print(mean_rep.size(), first_rep.size(), last_rep.size())
        return (mean_rep.detach().numpy(), first_rep.detach().numpy(), last_rep.detach().numpy() )

    mean_method_emb, first_method_emb, last_method_emb = [],[],[]
    mean_method_emb_test, first_method_emb_test, last_method_emb_test = [],[],[]
    
    for i in range (len(method_contents)): 
        mean_ , first_, last_ = encode_codebert_embedding(method_contents[i])
        mean_method_emb.append (mean_)
        first_method_emb.append (first_)
        last_method_emb.append (last_)
    

    mean_method_emb, first_method_emb, last_method_emb = np.array(mean_method_emb), np.array(first_method_emb), np.array(last_method_emb)
    print(  np.shape(mean_method_emb),  np.shape(first_method_emb), np.shape(last_method_emb))
    
    # save embeddings in cache
    pickle.dump( (mean_method_emb, first_method_emb, last_method_emb), open('./cache/'+args.project+'_all.pkl', 'wb'))


label_vocab = {"open":0, 'close':1}
y_ = all_df["category"]
X_ = np.array(all_df.drop(['category'], axis=1))
y_ = np.array([label_vocab[d] for d in y_])



if args.only_bert:
    if args.codebert_pooling == 'mean':
        X_  = mean_method_emb
    elif args.codebert_pooling == 'first':
        X_  = first_method_emb
    elif args.codebert_pooling == 'last':
        X_  = last_method_emb 

elif not args.only_SE:
    if args.codebert_pooling == 'mean':
        X_  = np.concatenate((X_, mean_method_emb), axis=1)
    elif args.codebert_pooling == 'first':
        X_  = np.concatenate((X_, first_method_emb), axis=1)
    elif args.codebert_pooling == 'last':
        X_  = np.concatenate((X_, last_method_emb), axis=1)
     
else:
  pass
   

if args.scaler == 'standard':
   scaler = StandardScaler()
elif args.scaler == 'minmax': 
   scaler = MinMaxScaler()
elif args.scaler == 'maxabs':
   scaler = MaxAbsScaler()
elif args.scaler == 'quantile':
   scaler = QuantileTransformer()
else:
   pass

if not args.scaler == 'none':
   X_ = scaler.fit_transform(X_)

X_train = X_[0:train_samples, :]
y_train = y_[0:train_samples]
X_test = X_[train_samples:, :]
y_test = y_[train_samples:]


#print(np.shape(X_), y_, len(y_))
#print(np.shape(X_train), np.shape(X_test))
#print(np.shape(y_train), np.shape(y_test), y_test)
#print(train_samples, test_samples)


def balance_pos_neg_in_training(X_train,y_train,balance):
    if balance == 'downsample':
       rus = RandomUnderSampler()
       X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elif balance == 'upsample':  
       ros = RandomOverSampler()
       X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    elif balance == 'smote':
       sm = SMOTE(k_neighbors=3)
       X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    else:
       #none balance
       X_resampled, y_resampled = X_train, y_train
    return X_resampled, y_resampled

def train_valid_model (model_name, X_train, X_test, y_train, y_test):
        #print(np.shape(X_train), y_train)
        #print(np.shape(X_test), y_test)
        if model_name == 'forest':
            model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        if model_name == 'tree':
            model = DecisionTreeClassifier().fit(X_train, y_train)
        if model_name == 'linear':
           model = LogisticRegression(max_iter=7000).fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        
      
        def evaluation_metrics(y_true, y_pred):
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
            auc_ = auc(fpr, tpr)
            y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
            print('predictions:', np.array(y_pred))
            acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            prc = precision_score(y_true=y_true, y_pred=y_pred)
            rc = recall_score(y_true=y_true, y_pred=y_pred)
            f1 = f1_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            far = fp/(fp+tn)
            return acc, prc, rc, f1, auc_, far
        
        acc, prc, rc, f1, auc_, far = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        return acc, prc, rc, f1, auc_, far


print('groundtruth:', y_test)
X_train, y_train = balance_pos_neg_in_training(X_train,y_train, args.balance)
acc1, prc1, rc1, f1_1, auc_1, far_1 = train_valid_model (args.model_name, X_train, X_test, y_train, y_test)
print('### Considered projects:', args.project)
print('### Model classifier:', args.model_name)
print('### Only CodeBERT:', args.only_bert)
print('### Only SE:', args.only_SE)
print('### Scaler used:', args.scaler)
print('### Balanced Stratigy used:', args.balance)
print('Acc:' , acc1, ' Prec:', prc1, ' Recal:', rc1, ' F1:', f1_1, ' AUC:', auc_1, 'FAR:', far_1 )

