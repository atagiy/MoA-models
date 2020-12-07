#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append('..')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[2]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
import copy
import seaborn as sn

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline,make_union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gc

import warnings
warnings.filterwarnings('ignore')


# In[3]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
              
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            else:
                props[col] = props[col].astype(np.float32)
            
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

def train_short_form_loader(feature_file,target_file,extra_target_file=None):
    '''takes the original target and features and creates a train dataset 
    in col long format'''


    train_features = pd.read_csv(feature_file)

    train_targets = pd.read_csv(target_file)
    train_features,_= reduce_mem_usage(train_features)
    train_targets,_ = reduce_mem_usage(train_targets)


    if extra_target_file is not None:
        extra_targets = pd.read_csv(extra_target_file)
        extra_targets,_ = reduce_mem_usage(extra_targets)
        train_targets = pd.merge(train_targets,extra_targets,on ='sig_id')
        del extra_targets

    targets = train_targets.columns[1:]

    train_melt=train_targets.merge(train_features,how="left",on="sig_id")


    del train_features,train_targets


    train_melt.set_index("sig_id",inplace=True)
    train_melt["cp_type"]= train_melt["cp_type"].astype('category')
    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')

    return train_melt , targets.to_list()



def test_short_form_loader(feature_file):
    '''takes the original target and features and creates a train dataset 
    in col long format'''


    train_features = pd.read_csv(feature_file)
    train_features,_= reduce_mem_usage(train_features)
    train_melt =  train_features.copy()
    del train_features


    train_melt.set_index("sig_id",inplace=True)
    train_melt["cp_type"]= train_melt["cp_type"].astype('category')
    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')

    return train_melt 


# In[4]:


input_directory = '../input/lish-moa/'

train,target_cols = train_short_form_loader(input_directory +'train_features.csv',input_directory+'train_targets_scored.csv')
test = test_short_form_loader(input_directory +"test_features.csv")

GENES = [col for col in train.columns if col.startswith('g-')]
CELLS = [col for col in train.columns if col.startswith('c-')]

from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import make_column_transformer,ColumnTransformer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# In[5]:


from sklearn.base import BaseEstimator,TransformerMixin

class CatIntMapper( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self ,col,dicti):
        self.col = col
        self.dicti = dicti
        
    def fit(self, X, y = None):
        
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
        assert  X[self.col].isin(self.dicti.keys()).all() 
        
       
        return pd.concat([X.drop(self.col,axis=1),X[self.col].map(self.dicti).astype(int).rename(self.col)],axis=1) 
    
    def transform( self, X):
        assert  X[self.col].isin(self.dicti.keys()).all() 
        
        return pd.concat([X.drop(self.col,axis=1),X[self.col].map(self.dicti).astype(int).rename(self.col)],axis=1) 

class NamedOutTWrapper( BaseEstimator, TransformerMixin ):
    
    def __init__(self,transformer,columns,inplace=False,prefix='_' ):
        
        self.transformer = transformer
        self.cols = columns
        self.inplace =  inplace
        self.prefix = prefix
        self.transformer_name = self._get_transformer_name()
        
    def fit(self, X, y = None):
            
        self.transformer =   self.transformer.fit(X[self.cols] , y )
            
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
       
        
        transformed_columns = self.transformer.fit_transform(X[self.cols] , y )
        out=pd.DataFrame(index=X.index)
        
       
        if self.inplace:
            out = X[self.cols]
            out[self.cols] = transformed_columns
            
            return pd.concat([X.drop(self.cols,axis=1),out],axis=1)
        else:
           
            for i,values in enumerate(transformed_columns.transpose()):
            
                out[ self.transformer_name + self.prefix + str(i)] = values
        
       
        
            return   pd.concat([X,out],axis=1)
    
    def transform( self, X):
        
        transformed_columns = self.transformer.transform(X[self.cols]  )
        
        out=pd.DataFrame(index=X.index)
        
        if self.inplace:
            out = X[self.cols]
            out[self.cols] = transformed_columns
            
            return pd.concat([X.drop(self.cols,axis=1),out],axis=1)
        else:
            for i,values in enumerate(transformed_columns.transpose()):

                out[ self.transformer_name + self.prefix + str(i)] = values

             
        return   pd.concat([X,out],axis=1)
            
    
    def _get_transformer_name(self):
        return str(self.transformer.__class__).split('.')[-1][0:-2]


class IdentityTransformer:
    '''Duummy_tansformer as a filler'''
    def __init__(self ):
        pass
    def fit(self, X, y = None):
        
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
        return  X
      
    
    def transform( self, X):
       
        return  X    

class SuppressControls( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self ):
        pass
    def fit(self, X, y = None):
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
      
        
        return   X.loc[X['cp_type']!='ctl_vehicle'].drop('cp_type', axis=1) 
    
    def transform( self, X):
       
       
        return    X.loc[X['cp_type']!='ctl_vehicle'].drop('cp_type', axis=1)


# In[6]:


def multifold_indexer(train,target_columns,n_splits=10,random_state=12347,**kwargs):
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits,random_state=random_state,**kwargs)
    folds[ 'kfold']=0
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=train[target_columns])):
        folds.iloc[v_idx,-1] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    return folds


# In[7]:


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    

    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    


# In[8]:


class DAE_Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size=1100,hidden_size2=1300):
        super(DAE_Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size2))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, hidden_size))

        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_features))

        
    def forward(self, x,mode='DAE'):

        x1 = F.relu(self.dense1(x))                  
        x2 = self.batch_norm2(x1)        
        x2 = F.relu(self.dense2(x2))        
        x3 = self.batch_norm3(x2)      
        x3 = F.relu(self.dense3(x3))        
        out = self.dense4(x3)
        
        if mode == 'DAE':
            return out
        else:
            return x1,x2,x3
    


# In[9]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline,make_union



map_controls = CatIntMapper('cp_type',{'ctl_vehicle': 0, 'trt_cp': 1})    

map_dose = CatIntMapper('cp_dose',{'D1': 1, 'D2': 0})    
map_time = CatIntMapper('cp_time',{24: 0, 48: 1, 72: 2})    


train = pd.read_csv(f'{input_directory}/train_features.csv')

GENES = [col for col in train.columns if col.startswith('g-')]
CELLS = [col for col in train.columns if col.startswith('c-')]

GENES

Rankg_g_tansform =  NamedOutTWrapper( QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),columns= GENES+CELLS,inplace=True)

PCA_g_tansform =  NamedOutTWrapper(PCA(20),columns= GENES,prefix ='_g' )

PCA_c_tansform =  NamedOutTWrapper(PCA(20),columns= CELLS,prefix ='_c' )



from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, cols ):
        self.cols=cols
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):

        return X.drop(self.cols,axis=1)



CatDropper =ColumnDropper(cols=['cp_type','cp_time','cp_dose'])

transformers_list=[map_controls,map_dose,map_time,Rankg_g_tansform,CatDropper]


# In[10]:


#exp_name = 'test_DAE_0.2_all_together'
exp_name = 'test_DAE_all_together'

def run_inference(X_train,y_train,X_valid,y_valid,X_test,fold, seed,inference_only=False,**kwargs):
    seed_everything(seed)
    if not  inference_only:
        train_dataset = MoADataset(X_train, y_train)
        valid_dataset = MoADataset(X_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    testdataset = TestDataset(X_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DAE_Model(
        num_features= X_train.shape[1] ,
        num_targets=  X_train.shape[1],
       # hidden_size=hidden_size,
        **kwargs
    )
    
    
    model.load_state_dict(torch.load( f"../input/pytorch-swap-denoisingautoencoder/FOLD{fold}_{exp_name}.pth",map_location=torch.device('cpu')))
    
    model.to(DEVICE)
    
    if not  inference_only:
        oof = inference_infer_features_fn(model, validloader, DEVICE)    
    else:
        oof= 0
    

    predictions = infer_features_fn(model, testloader, DEVICE)
    
    predictions = predictions
    
    return oof, predictions

transformers_list


DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1000
BATCH_SIZE = 640
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-8
NFOLDS = 10
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
GAMMA=0.5
FACTOR=0.75

hidden_size=1100
hidden_size2=1300
PATIENCE=10
THRESHOLD = 5e-3


def infer_features_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs,mode='get_features')
      
        preds.append(torch.cat(outputs,axis=1).detach().cpu().numpy())
        
        
        
    preds = np.concatenate(preds)
    
    return preds
   
 


# In[11]:


SEED = [0]
train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')
test = test_short_form_loader("../input/lish-moa/test_features.csv")

train = pd.concat([train,test])
train[target_cols]= train[target_cols].fillna(0)
test = train.copy()


oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
 
 train = multifold_indexer(train,target_cols,n_splits=NFOLDS)

 for fold in range(NFOLDS):

 
     train_df = train[train['kfold'] != fold]
     valid_df = train[train['kfold'] == fold]

     
     feature_cols = [col  for col in train_df.columns if not (col in target_cols+['kfold'])]
     
     pipeline_val = make_pipeline(*transformers_list)
     
     X_train, y_train  = train_df[feature_cols], train_df[target_cols]
     X_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
     
     X_train = pipeline_val.fit_transform(X_train)
     
     X_train = X_train.values
          
     X_valid = pipeline_val.transform(X_valid)
          
     valid_index = X_valid.index
    
     X_valid = X_valid.values
     
     y_train = y_train.values
  
     X_test = test[feature_cols]
  
     X_test = pipeline_val.transform(X_test).values

    
     pred_ = run_inference(X_train,y_train,X_valid,y_valid,X_test,fold, seed,inference_only=True)    
     
     break


# In[12]:


transformed_features = pd.DataFrame(pred_[1],index=test.index)


# In[13]:


class DAE_Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size=1100,hidden_size2=1300):
        super(DAE_Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))    
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size2))      
        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, hidden_size))
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_features))

        
    def forward(self, x,mode='DAE'):

        x1 = F.relu(self.dense1(x))
        
            
        x2 = self.batch_norm2(x1)

        x2 = F.relu(self.dense2(x2))
        
        x3 = self.batch_norm3(x2)
      
        x3 = F.relu(self.dense3(x3))
        
        out = self.dense4(x3)
        
        if mode == 'DAE':
            return out
        else:
            return x1,x2,x3
    


# In[14]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline,make_union

map_controls = CatIntMapper('cp_type',{'ctl_vehicle': 0, 'trt_cp': 1})    

map_dose = CatIntMapper('cp_dose',{'D1': 1, 'D2': 0})    
map_time = CatIntMapper('cp_time',{24: 0, 48: 1, 72: 2})    

train = pd.read_csv(f'{input_directory}/train_features.csv')

GENES = [col for col in train.columns if col.startswith('g-')]
CELLS = [col for col in train.columns if col.startswith('c-')]

Rankg_g_tansform =  NamedOutTWrapper( QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),columns= GENES+CELLS,inplace=True)

PCA_g_tansform =  NamedOutTWrapper(PCA(20),columns= GENES,prefix ='_g' )

PCA_c_tansform =  NamedOutTWrapper(PCA(20),columns= CELLS,prefix ='_c' )

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, cols ):
        self.cols=cols
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):

        return X.drop(self.cols,axis=1)


# In[15]:



CatDropper =ColumnDropper(cols=['cp_type','cp_time','cp_dose'])

transformers_list=[map_controls,map_dose,map_time,Rankg_g_tansform,CatDropper]

#exp_name = 'test_DAE_0.2_all_together'
exp_name = 'test_DAE_all_together'


# In[16]:


def run_inference(X_train,y_train,X_valid,y_valid,X_test,fold, seed,inference_only=False,**kwargs):
    seed_everything(seed)
    if not  inference_only:
        train_dataset = MoADataset(X_train, y_train)
        valid_dataset = MoADataset(X_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    testdataset = TestDataset(X_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DAE_Model(
        num_features= X_train.shape[1] ,
        num_targets=  X_train.shape[1],
       # hidden_size=hidden_size,
        **kwargs
    )
    
    
    model.load_state_dict(torch.load( f"../input/pytorch-swap-denoisingautoencoder/FOLD{fold}_{exp_name}.pth",map_location=torch.device('cpu')))#map_location='cuda:0'))#,freeze_first_layer=True)
    
    model.to(DEVICE)
    
    if not  inference_only:
        oof = inference_infer_features_fn(model, validloader, DEVICE)    
    else:
        oof= 0
    

    predictions = infer_features_fn(model, testloader, DEVICE)
    
    predictions = predictions
    
    return oof, predictions


# In[17]:



DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1000
BATCH_SIZE = 640
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-8
NFOLDS = 10
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
GAMMA=0.5
FACTOR=0.75

hidden_size=1100
hidden_size2=1300
PATIENCE=10
THRESHOLD = 5e-3


# In[18]:


def infer_features_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs,mode='get_features')
        
        
        preds.append(torch.cat(outputs,axis=1).detach().cpu().numpy())
        
        
        
    preds = np.concatenate(preds)
    
    return preds
   
    


# In[19]:


#SEED = [0,12347,565657,123123,78591]
SEED = [0]
train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')
test = test_short_form_loader("../input/lish-moa/test_features.csv")

train = pd.concat([train,test])
train[target_cols]= train[target_cols].fillna(0)
test = train.copy()


oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
    
    train = multifold_indexer(train,target_cols,n_splits=NFOLDS)

    for fold in range(NFOLDS):

    
        train_df = train[train['kfold'] != fold]
        valid_df = train[train['kfold'] == fold]

        
        feature_cols = [col  for col in train_df.columns if not (col in target_cols+['kfold'])]

        
        pipeline_val = make_pipeline(*transformers_list)
        
        X_train, y_train  = train_df[feature_cols], train_df[target_cols]
        X_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
        

        X_train = pipeline_val.fit_transform(X_train)        
        
        X_train = X_train.values
        
        
        X_valid = pipeline_val.transform(X_valid)
        
        
        
        
        valid_index = X_valid.index
        X_valid = X_valid.values
        
        y_train = y_train.values
        
        X_test = test[feature_cols]
                    
        X_test = pipeline_val.transform(X_test).values
              
        pred_ = run_inference(X_train,y_train,X_valid,y_valid,X_test,fold, seed,inference_only=True)    
        
        break


# In[20]:


transformed_features = pd.DataFrame(pred_[1],index=test.index)


# In[21]:


transformed_features.columns = [str(i) for i in range(len(transformed_features.columns))]

transformed_features.reset_index().to_feather('./features_0.2_altogether.fth')


# In[22]:


import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.base import BaseEstimator, TransformerMixin


# In[23]:


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    


# In[24]:


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    


# In[25]:


class DaeAdder( BaseEstimator, TransformerMixin ):
    def __init__(self,filename):
        
        self.filename=filename
    
    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,X,y=None):
        
        Dae_features = pd.read_feather(self.filename).set_index('sig_id')
        
        return X.merge(Dae_features,how='left', on='sig_id')
        
        
    def transform(self,X):
        
        Dae_features = pd.read_feather(self.filename).set_index('sig_id')
        
        return X.merge(Dae_features,how='left', on='sig_id')
        
        


# In[26]:


train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')

Dae0_2 =DaeAdder(filename='features_0.2_altogether.fth')

#res = Dae0_2.fit_transform(train)


# In[27]:


class SupressControls( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self ):
        pass
    def fit(self, X, y = None):
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
        X = X[X['cp_type']!=0]
        X = X.drop('cp_type', axis=1)
        return X 
    
    def transform( self, X):
        X = X[X['cp_type']!=0]
        X = X.drop('cp_type', axis=1)
        return X
    
class CatIntMapper( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self ,col,dicti):
        self.col = col
        self.dicti = dicti
        
    def fit(self, X, y = None):
        
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
        assert  X[self.col].isin(self.dicti.keys()).all() 
        
       
        return pd.concat([X.drop(self.col,axis=1),X[self.col].map(self.dicti).astype(int).rename(self.col)],axis=1) 
    
    def transform( self, X):
        assert  X[self.col].isin(self.dicti.keys()).all() 
        
        return pd.concat([X.drop(self.col,axis=1),X[self.col].map(self.dicti).astype(int).rename(self.col)],axis=1) 

class NamedOutTWrapper( BaseEstimator, TransformerMixin ):
    
    def __init__(self,transformer,columns,inplace=False,prefix='_' ):
        
        self.transformer = transformer
        self.cols = columns
        self.inplace =  inplace
        self.prefix = prefix
        self.transformer_name = self._get_transformer_name()
        
    def fit(self, X, y = None):
            
        self.transformer =   self.transformer.fit(X[self.cols] , y )
            
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
       
        
        transformed_columns = self.transformer.fit_transform(X[self.cols] , y )
        out=pd.DataFrame(index=X.index)
        
       
        if self.inplace:
            out = X[self.cols].copy()
            out[self.cols] = transformed_columns
            
            return pd.concat([X.drop(self.cols,axis=1),out],axis=1)
        else:
           
            for i,values in enumerate(transformed_columns.transpose()):
            
                out[ self.transformer_name + self.prefix + str(i)] = values
        
       
        
            return   pd.concat([X,out],axis=1)
    
    def transform( self, X):
        
        transformed_columns = self.transformer.transform(X[self.cols]  )
        
        out=pd.DataFrame(index=X.index)
        
        if self.inplace:
            out = X[self.cols].copy()
            out[self.cols] = transformed_columns
            
            return pd.concat([X.drop(self.cols,axis=1),out],axis=1)
        else:
            for i,values in enumerate(transformed_columns.transpose()):

                out[ self.transformer_name + self.prefix + str(i)] = values

             
        return   pd.concat([X,out],axis=1)
            
    
    def _get_transformer_name(self):
        return str(self.transformer.__class__).split('.')[-1][0:-2]


class IdentityTransformer:
    '''Duummy_tansformer as a filler'''
    def __init__(self ):
        pass
    def fit(self, X, y = None):
        
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
        return  X
      
    
    def transform( self, X):
       
        return  X    

class SuppressControls( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self ):
        pass
    def fit(self, X, y = None):
        return self
    #Return self nothing else to do here
    def fit_transform( self, X, y = None  ):
        
      
        
        return   X.loc[X['cp_type']=='trt_cp'].drop('cp_type', axis=1) 
    
    def transform( self, X):
       
       
        return    X.loc[X['cp_type']=='trt_cp'].drop('cp_type', axis=1)


class ColumnDropper( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, cols ):
        self.cols=cols
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):

        return X.drop(self.cols,axis=1)


from sklearn.base import BaseEstimator,TransformerMixin
#Custom transformer that breaks dates column into year, month and day into separate columns and
#converts certain features to binary 
class VarianceFilter( BaseEstimator, TransformerMixin ):
    def __init__(self,threshold):
        self.threshold = threshold
    def fit(self,X,y=None):
        mask = X.var()<= self.threshold
        self.drop_cols = set([ col for val,col in zip(mask,X.columns) if val])
        self.drop_cols.discard('cp_type')
        return self
    def transform(self,X):
        
        return X.drop(self.drop_cols,axis=1)


def apply_pipe_together(pipeline,train,test):
    #@add warning when intesection is not the whole
    data = pd.concat([train,test])

    data = pipeline.fit_transform(data)
    
    train = data.loc[data.index.intersection(train.index)]
    test = data.loc[data.index.intersection(test.index)]
    
    return pipeline,train,test


# In[28]:



map_controls = CatIntMapper('cp_type',{'ctl_vehicle': 0, 'trt_cp': 1})    

map_dose = CatIntMapper('cp_dose',{'D1': 1, 'D2': 0})    
map_time = CatIntMapper('cp_time',{24: 0, 48: 1, 72: 2})    


# In[29]:


from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[30]:


from sklearn.model_selection import GroupKFold
def groupkfold_indexer(train,target_columns,n_splits=10,random_state=12347,**kwargs):
    #@alternatively use groushufflesplit to obtein different groups
    
    
    folds = train.copy()

    groups =pd.read_csv("input/aggclus_taco/aggclus3_results.csv",index_col='sig_id')['aggclus1-clusters']
    
    gkf = GroupKFold(n_splits=n_splits)
    folds[ 'kfold']=0
    for f, (t_idx, v_idx) in enumerate(gkf.split(X=train.sample(len(train),random_state=random_state), y=train[target_columns].sample(len(train),random_state=random_state),groups= groups.loc[train.index].sample(len(train),random_state=random_state) )):
        folds.iloc[v_idx,-1] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    return folds


# In[31]:


def multifold_indexer(train,target_columns,n_splits=10,random_state=12347,**kwargs):
    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits,random_state=random_state,**kwargs)
    folds[ 'kfold']=0
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=train[target_columns])):
        folds.iloc[v_idx,-1] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    return folds


# In[32]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size1=388,hidden_size2=512,drop_rate1=0.0,drop_rate2=0.3,drop_rate3=0.3):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(drop_rate1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size1))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(drop_rate2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size1, hidden_size2))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(drop_rate3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, num_targets))


    def forward(self, x):
        
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# In[33]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if not  scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, scheduler, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    if scheduler.__class__ ==  torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(final_loss)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
   
    


# In[34]:


exp_name = 'test_DAE_0.2_all_together'

def run_training(X_train,y_train,X_valid,y_valid,X_test,fold, seed,verbose=False,**kwargs):
    
    seed_everything(seed)
    
   
    
    train_dataset = MoADataset(X_train, y_train)
    valid_dataset = MoADataset(X_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features= X_train.shape[1] ,
        num_targets=  y_train.shape[1],hidden_size1=hidden_size1,hidden_size2=hidden_size2,
       **kwargs
    )
    
    
    model.to(DEVICE)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
    
    loss_val = nn.BCEWithLogitsLoss()

    loss_tr = SmoothBCEwLogits(smoothing =0.001)
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    best_loss = np.inf
    
    
    
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
        if verbose:
            print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model,scheduler, loss_val, validloader, DEVICE)
        if verbose:
            print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof = valid_preds

            ##torch.save(model.state_dict(), f"{exp_name}_SEED{seed}_FOLD{fold}.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    ##torch.save(model.state_dict(), f"{exp_name}_SEED{seed}_FOLD{fold}.pth")
    #--------------------- PREDICTION---------------------
   
    testdataset = TestDataset(X_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    model = Model(
          num_features= X_train.shape[1] ,
         num_targets=  y_train.shape[1],
         ##hidden_size=hidden_size,
        hidden_size1=hidden_size1,hidden_size2=hidden_size2,
        **kwargs
     )
    
    ##model.load_state_dict(torch.load(f"../results/FOLD{fold}_{exp_name}.pth"))
    model.load_state_dict(torch.load( f"../input/auto-encoder2/{exp_name}_SEED{seed}_FOLD{fold}.pth",map_location=torch.device('cpu')))
    model.to(DEVICE)
    
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[35]:


#params for one cycle schedule
DEVICE =  torch.device('cuda:0')
EPOCHS = 1 #70
BATCH_SIZE = 1024
LEARNING_RATE = 6e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 7              
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False #False


hidden_size1=2048

hidden_size2=2048

transformers_list=[map_dose,map_time,Dae0_2,SuppressControls()]


# In[36]:


def run_k_fold(folds,target_cols,test,transformers_list,NFOLDS, seed,verbose=False,**kwargs):
    
    
    train = folds
    test_ = test
    
    oof = train[target_cols].copy()
    oof = oof*0
    predictions = pd.DataFrame(0,columns=target_cols,index=test.index)
    
    #print(test_.head())
    for fold in range(NFOLDS):
        
    
        train_df = train[train['kfold'] != fold]
        valid_df = train[train['kfold'] == fold]
        
        
        feature_cols = [col  for col in train_df.columns if not (col in target_cols+['kfold'])]
        
        
        pipeline_val = make_pipeline(*transformers_list)
        
        X_train, y_train  = train_df[feature_cols], train_df[target_cols]
        X_valid, y_valid =  valid_df[feature_cols], valid_df[target_cols].values
        
      
       
        X_train = pipeline_val.fit_transform(X_train,y_train)
        feature_cols = [col  for col in X_train.columns if not (col in target_cols+['kfold'])]
        
        X_train = X_train.values
        
        
        X_valid = pipeline_val.transform(X_valid)
        valid_index = X_valid.index
        X_valid = X_valid.values
        
        y_train = y_train.values
        
        X_test = pipeline_val.transform(test_)
        test_index = X_test.index
        X_test = X_test[feature_cols].values
            
        oof_, pred_ = run_training(X_train,y_train,X_valid,y_valid,X_test,fold, seed,verbose,**kwargs)
        
        
        oof.loc[valid_index] = oof_
        
        
        
        predictions.loc[test_index] += pred_ / NFOLDS
        
        
    return oof, predictions


# In[37]:


params={}

# Averaging on multiple SEEDS
SEED = [0]
train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')
test = test_short_form_loader("../input/lish-moa/test_features.csv")


pipeline_test = make_pipeline(*transformers_list)
pipeline_test,train , test = apply_pipe_together(pipeline_test,train,test)

transformers_list=[IdentityTransformer()]

oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
   
    folds = multifold_indexer(train,target_cols,n_splits=NFOLDS)
    oof_, predictions_ = run_k_fold(folds,target_cols,test,transformers_list,NFOLDS, seed,verbose=True,**params)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)

test[target_cols] = predictions


# In[38]:



y_true = train[target_cols].values
y_pred = oof

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred.iloc[:, i],labels=[0,1])
    score +=( score_ / len(target_cols))
    
#print("CV log_loss: ", score)
    


# In[39]:


train,target_cols = train_short_form_loader('../input/lish-moa/train_features.csv','../input/lish-moa/train_targets_scored.csv')
y_true = train[target_cols].values

y_pred = train[target_cols].copy()
y_pred[target_cols] = 0
y_pred.loc[oof.index] = oof
y_pred.loc[train.cp_type=='ctl_vehicle'] = 0

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred.iloc[:, i],labels=[0,1])
    score +=( score_ / len(target_cols))
    
#print("CV log_loss: ", score)
    


# In[40]:


sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission.set_index('sig_id',inplace=True)
test_features.set_index('sig_id',inplace=True)
test_features = test_features.loc[sample_submission.index]

sub4 = sample_submission.drop(columns=target_cols).merge(test[target_cols], on='sig_id', how='left').fillna(0)
sub4.loc[test_features['cp_type']=='ctl_vehicle', target_cols] =0
sub4.to_csv('sub4.csv', index=True)


# In[41]:


del GENES
del CELLS
del train
del test
del y_pred
del y_true
del target_cols
del sample_submission
del predictions
del oof
del transformed_features
del Dae0_2


# In[42]:


gc.collect()


# In[ ]:




