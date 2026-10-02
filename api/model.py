import random
import os
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from io import StringIO
from unicodedata import category
from markdown import markdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification, BertForMaskedLM,
    XLNetTokenizer, XLNetForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM,
    AlbertTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM,
    get_scheduler
)
from torch.optim import AdamW

MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # For NVIDIA GPUs (rare on macOS)
else:
    device = torch.device("cpu")   # Fallback to CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ['bert', 'xlnet', 'roberta', 'albert']

MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'),
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
        ]

def train_model(train_df, model_save_path, model_task, model_tokenizer, model_name):
    """
    Trains a sentiment classification model on the provided dataset.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data with 'text' and 'polarity' columns.
        model_save_path (str): Path to save the best model.
        model_task (class): Transformer model class, e.g., AutoModelForSequenceClassification
        model_tokenizer (class): Tokenizer class, e.g., AutoTokenizer
        model_name (str): Hugging Face model name, e.g., "bert-base-cased"

    Returns:
        str: The path where the best model was saved.
    """
    seed_torch(42)


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    
    tokenizer = model_tokenizer.from_pretrained(model_name, do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

    input_ids, attention_masks = [], []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print(f'Training data shape: {input_ids.shape}, {attention_masks.shape}, {labels.shape}')

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42
    )
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.1, random_state=42
    )

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    # Use passed-in model_task and model_name
    model = model_task.from_pretrained(model_name, num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    print(f"Using device: {device}")

    print("Starting training...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

        train_acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")

        model.eval()
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
                val_labels.extend(b_labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")

    print("Final Model Performance on Validation Set:")
    print(classification_report(val_labels, val_predictions, digits=4))
    return model_save_path

def predict_model(predict_df, model_saved_path, model_task, model_tokenizer, model_name):
    """
    Uses a pre-trained transformer model (BERT, XLNet, RoBERTa, etc.) 
    for sentiment classification and adds a polarity column in the test dataset 
    with predicted values.

    Args:
    - predict_df (pd.DataFrame): DataFrame containing text data to classify.
    - model_saved_path (str): Path to the saved model.
    - model_task (class): e.g., BertForSequenceClassification, XLNetForSequenceClassification
    - model_tokenizer (class): e.g., BertTokenizer, XLNetTokenizer
    - model_name (str): e.g., "bert-base-cased"

    Returns:
    pd.DataFrame: Same DataFrame with a new 'polarity' column containing predicted labels.
    """
    
    seed_torch(42)

    tokenizer = model_tokenizer.from_pretrained(model_name, do_lower_case=True)

    begin = time.time()
    sentences = predict_df.text.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    # Instantiate model dynamically
    model = model_task.from_pretrained(model_name, num_labels=3)
    model.load_state_dict(torch.load(model_saved_path))
    model.eval()

    predictions = []
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        predictions.append(logits.detach().cpu().numpy())

    end = time.time()
    print(f'Prediction used {end - begin:.2f} seconds')

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Add polarity column
    predict_df['polarity'] = flat_predictions

    return predict_df

"""
def train_model(train_df, model_save_path, model_select=0):
  
Trains a sentiment classification model on the provided dataset.

Args:
- train_df (pd.DataFrame)
- model_save_path (str)
- model_select (int, optional)

Returns:
        str: The path where the best model was saved.
    
Notes:
        - Converts sentiment labels to numeric form (positive=1, negative=2, neutral=0).
        - Saves the models
    
    seed_torch(42)

    cur_model = MODELS[model_select]
    m_name = MODEL_NAMES[model_select]


    train_df['Polarity'] = train_df['Polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.Text.values
    labels = train_df.Polarity.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            str(sent),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print(f'Training data shape: {input_ids.shape}, {attention_masks.shape}, {labels.shape}')


    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42)
    train_masks, val_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.1, random_state=42)


    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)


    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    print("Starting training...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

        train_acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {train_acc:.4f}")


        model.eval()
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
                val_labels.extend(b_labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")


        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")


    print("Final Model Performance on Validation Set:")
    print(classification_report(val_labels, val_predictions, digits=4))
    return model_save_path
"""

def seed_torch(seed):
    """
Set random seeds for reproducibility in PyTorch and related libraries. 

Args: 
- Seed (int) : number to use for all random generators. 

Example:
seed_torch(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

"""
def test_model(test_df, model_saved_path, model_select=0):
  
Tests a pre-trained sentiment classification model on a test dataset and evaluates its performance.
    
Args:
- test_df (pd.DataFrame)
- model_saved_path (str)
- model_select (int, optional)
    
Returns:
pd.DataFrame: A DataFrame with the original test data and the model's predictions.

  MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'),
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
        ]
  MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
  seed_torch(42)

  cur_model=MODELS[model_select]
  m_name=MODEL_NAMES[model_select]

  tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

  begin=time.time()

  test_df['Polarity']=test_df['Polarity'].replace({
      'positive':1,
      'negative':2,
      'neutral':0})


  sentences = test_df.Text.values
  labels = test_df.Polarity.values

  input_ids = []
  attention_masks = []

  for sent in sentences:
      encoded_dict = tokenizer.encode_plus(
                          str(sent),
                          add_special_tokens = True,
                          max_length = MAX_LEN,
                          pad_to_max_length = True,
                          return_attention_mask = True,
                          return_tensors = 'pt',
                    )

      input_ids.append(encoded_dict['input_ids'])
      attention_masks.append(encoded_dict['attention_mask'])

  prediction_inputs = torch.cat(input_ids,dim=0)
  prediction_masks = torch.cat(attention_masks,dim=0)
  prediction_labels = torch.tensor(labels)

  prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

  model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
  model.load_state_dict(torch.load(model_saved_path))
# model.cuda()
  model.eval()

  predictions,true_labels=[],[]

  for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch

      with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits = outputs[0]

      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      predictions.append(logits)
      true_labels.append(label_ids)

  end=time.time()
  print('Prediction used {:.2f} seconds'.format(end - begin))

  flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  flat_true_labels = [item for sublist in true_labels for item in sublist]

  print("Accuracy of {} is: {}".format(m_name, accuracy_score(flat_true_labels,flat_predictions)))

  print(classification_report(flat_true_labels,flat_predictions))


  df_prediction = pd.DataFrame(flat_predictions, columns=['prediction_Polarity'])

  df_combined = pd.concat([test_df, df_prediction], axis=1)

  counts = df_combined['prediction_Polarity'].value_counts()
  print(counts)

  return df_combined
"""