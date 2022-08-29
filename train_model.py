import json
import pandas as pd
import random
from tqdm import tqdm
import os
import yaml

import mlflow

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer,  AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch


# Load the preprocessed data
data_path = "./data/airline_sentiment_preprocessed_data.csv"

# Load the airline sentiment dataset
airline_data = pd.read_csv(data_path)

# Create train and validation data sets
# Create training and validation data
X_train, X_val, y_train, y_val = train_test_split(airline_data.index.values, 
                                                  airline_data.label.values, 
                                                  test_size = 0.15, 
                                                  random_state = 2022, 
                                                  stratify = airline_data.label.values)

# Create the data type columns
airline_data.loc[X_train, 'data_type'] = 'train'
airline_data.loc[X_val, 'data_type'] = 'val'


"""
Get model tokenizer and encode data
"""
# Get the FinBERT Tokenizer
finbert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode the Training and Validation Data
encoded_data_train = finbert_tokenizer.batch_encode_plus(
    airline_data[airline_data.data_type=='train'].text.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 # the maximum lenght observed in the headlines
)

encoded_data_val = finbert_tokenizer.batch_encode_plus(
    airline_data[airline_data.data_type=='val'].text.values, 
    return_tensors='pt',
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=150 # the maximum lenght observed in the headlines
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(airline_data[airline_data.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
sentiments_val = torch.tensor(airline_data[airline_data.data_type=='val'].label.values)


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, sentiments_val)


"""
Set Up the BERT pretrained model
"""
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=3,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

"""
Create Data Loaders
"""

batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


"""
Create Optimizer and Scheduler
"""
optimizer = AdamW(model.parameters(),lr=1e-5, eps=1e-8)
epochs = 1
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


"""
Evaluation metrics
"""
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


"""
Training loop
"""
seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    torch.save(model.state_dict(), f'finetuned_finBERT_epoch_{epoch}.model')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)


if __name__ == "__main__":


    mlflow_config = yaml.safe_load(open("config.yaml"))["credentials"]["mlflow_credentials"]
    metrics_file = "./metrics/metrics.json"

    MLFLOW_TRACKING_URI= mlflow_config['MLFLOW_TRACKING_URI']
    MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
    MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD'] 

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run():

        # Load the metrics file
        with open(metrics_file) as metrics_info:

            metrics_values = json.load(metrics_info)

            # Track them with MLFlow
            mlflow.log_metric("Precision", metrics_values['precision'])
            mlflow.log_metric("Recall", metrics_values['recall'])
            mlflow.log_metric("F1-Score", metrics_values['f1-score'])
