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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ['bert', 'xlnet', 'roberta', 'albert']

MODELS = [(BertForSequenceClassification,BertTokenizer,'bert-base-cased'),
          (XLNetForSequenceClassification, XLNetTokenizer,'xlnet-base-cased'),
          (RobertaForSequenceClassification, RobertaTokenizer,'roberta-base'),
          (AlbertForSequenceClassification, AlbertTokenizer,'albert-base-v1')
        ]

def train_model_bert(train_df, model_save_path):
    
    """
    Trains a Bert-based sentiment classification model on the provided dataset.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training data with 'text' and 'polarity' columns.
    - model_save_path (str): Path to save the best model.

    Returns:
    - str: The path where the best model was saved.

    Notes:
    - Converts sentiment labels to numeric form (positive=1, negative=2, neutral=0).
    - Saves the model with the highest F1 score on the validation set.
    """

    seed_torch(42)

    cur_model = MODELS[0]
    m_name = MODEL_NAMES[0]


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

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

def train_model_xlnet(train_df, model_save_path):

    """
    Trains an XLNet-based sentiment classification model on the provided dataset.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training data with 'text' and 'polarity' columns.
    - model_save_path (str): Path to save the best model.

    Returns:
    - str: The path where the best model was saved.

    Notes:
    - Converts sentiment labels to numeric form (positive=1, negative=2, neutral=0).
    - Saves the model with the highest F1 score on the validation set.
    """

    seed_torch(42)

    cur_model = MODELS[1]
    m_name = MODEL_NAMES[1]


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

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

def train_model_roberta(train_df, model_save_path):

    """
    Trains a Roberta-based sentiment classification model on the provided dataset.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training data with 'text' and 'polarity' columns.
    - model_save_path (str): Path to save the best model.

    Returns:
    - str: The path where the best model was saved.

    Notes:
    - Converts sentiment labels to numeric form (positive=1, negative=2, neutral=0).
    - Saves the model with the highest F1 score on the validation set.
    """
    
    seed_torch(42)

    cur_model = MODELS[2]
    m_name = MODEL_NAMES[2]


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

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

def train_model_albert(train_df, model_save_path):

    """
    Trains an Albert-based sentiment classification model on the provided dataset.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training data with 'text' and 'polarity' columns.
    - model_save_path (str): Path to save the best model.

    Returns:
    - str: The path where the best model was saved.

    Notes:
    - Converts sentiment labels to numeric form (positive=1, negative=2, neutral=0).
    - Saves the model with the highest F1 score on the validation set.
    """

    seed_torch(42)

    cur_model = MODELS[3]
    m_name = MODEL_NAMES[3]


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

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

def train_model(train_df, model_save_path, model_select=0):
    """
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
    """
    seed_torch(42)

    cur_model = MODELS[model_select]
    m_name = MODEL_NAMES[model_select]


    train_df['polarity'] = train_df['polarity'].replace({'positive': 1, 'negative': 2, 'neutral': 0})
    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    sentences = train_df.text.values
    labels = train_df.polarity.values

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

def test_model(test_df, model_saved_path, model_select=0):
  """
Tests a pre-trained sentiment classification model on a test dataset and evaluates its performance.
    
Args:
- test_df (pd.DataFrame)
- model_saved_path (str)
- model_select (int, optional)
    
Returns:
pd.DataFrame: A DataFrame with the original test data and the model's predictions.
    """

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

  test_df['polarity']=test_df['polarity'].replace({
      'positive':1,
      'negative':2,
      'neutral':0})


  sentences = test_df.text.values
  labels = test_df.polarity.values

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


  df_prediction = pd.DataFrame(flat_predictions, columns=['prediction_polarity'])

  df_combined = pd.concat([test_df, df_prediction], axis=1)

  counts = df_combined['prediction_polarity'].value_counts()
  print(counts)

  return df_combined

def predict_bert(test_df, model_saved_path):
    
    """
    Runs inference with a pre-trained BERT sentiment classification model and updates the
    polarity column in the test dataset with predicted values.

    Args:
    - test_df (pd.DataFrame): DataFrame containing text and original polarity labels.
    - model_saved_path (str): Path to the saved BERT model.

    Returns:
    pd.DataFrame: The same DataFrame with polarity replaced by BERT predictions.
    """

    MODELS = [
        (BertForSequenceClassification, BertTokenizer, 'bert-base-cased'),
        (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
        (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
        (AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1')
    ]
    MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
    seed_torch(42)

    cur_model = MODELS[0]
    m_name = MODEL_NAMES[0]

    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    begin = time.time()

    # Convert string labels to numbers only if needed
    test_df['polarity'] = test_df['polarity'].replace({
        'positive':1,
        'negative':2,
        'neutral':0
    })

    sentences = test_df.text.values
    labels = test_df.polarity.values

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
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.load_state_dict(torch.load(model_saved_path))
    model.eval()

    predictions = []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    end = time.time()
    print('Prediction used {:.2f} seconds'.format(end - begin))

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Overwrite polarity column with predicted values
    test_df['polarity'] = flat_predictions

    return test_df

def predict_xlnet(test_df, model_saved_path):

    """
    Runs inference with a pre-trained XLNet sentiment classification model and updates the
    polarity column in the test dataset with predicted values.

    Args:
    - test_df (pd.DataFrame): DataFrame containing text and original polarity labels.
    - model_saved_path (str): Path to the saved XLNet model.

    Returns:
    pd.DataFrame: The same DataFrame with polarity replaced by XLNet predictions.
    """

    MODELS = [
        (BertForSequenceClassification, BertTokenizer, 'bert-base-cased'),
        (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
        (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
        (AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1')
    ]
    MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
    seed_torch(42)

    cur_model = MODELS[1]
    m_name = MODEL_NAMES[1]

    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    begin = time.time()

    # Convert string labels to numbers only if needed
    test_df['polarity'] = test_df['polarity'].replace({
        'positive':1,
        'negative':2,
        'neutral':0
    })

    sentences = test_df.text.values
    labels = test_df.polarity.values

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
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.load_state_dict(torch.load(model_saved_path))
    model.eval()

    predictions = []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    end = time.time()
    print('Prediction used {:.2f} seconds'.format(end - begin))

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Overwrite polarity column with predicted values
    test_df['polarity'] = flat_predictions

    return test_df

def predict_roberta(test_df, model_saved_path):

    """
    Runs inference with a pre-trained RoBERTa sentiment classification model and updates the
    polarity column in the test dataset with predicted values.

    Args:
    - test_df (pd.DataFrame): DataFrame containing text and original polarity labels.
    - model_saved_path (str): Path to the saved RoBERTa model.

    Returns:
    pd.DataFrame: The same DataFrame with polarity replaced by RoBERTa predictions.
    """

    MODELS = [
        (BertForSequenceClassification, BertTokenizer, 'bert-base-cased'),
        (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
        (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
        (AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1')
    ]
    MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
    seed_torch(42)

    cur_model = MODELS[2]
    m_name = MODEL_NAMES[2]

    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    begin = time.time()

    # Convert string labels to numbers only if needed
    test_df['polarity'] = test_df['polarity'].replace({
        'positive':1,
        'negative':2,
        'neutral':0
    })

    sentences = test_df.text.values
    labels = test_df.polarity.values

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
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.load_state_dict(torch.load(model_saved_path))
    model.eval()

    predictions = []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    end = time.time()
    print('Prediction used {:.2f} seconds'.format(end - begin))

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Overwrite polarity column with predicted values
    test_df['polarity'] = flat_predictions

    return test_df

def predict_albert(test_df, model_saved_path):

    """
    Runs inference with a pre-trained ALBERT sentiment classification model and updates the
    polarity column in the test dataset with predicted values.

    Args:
    - test_df (pd.DataFrame): DataFrame containing text and original polarity labels.
    - model_saved_path (str): Path to the saved ALBERT model.

    Returns:
    pd.DataFrame: The same DataFrame with polarity replaced by ALBERT predictions.
    """

    MODELS = [
        (BertForSequenceClassification, BertTokenizer, 'bert-base-cased'),
        (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
        (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
        (AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v1')
    ]
    MODEL_NAMES = ['bert', 'xlnet', 'Roberta', 'albert']
    seed_torch(42)

    cur_model = MODELS[3]
    m_name = MODEL_NAMES[3]

    tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

    begin = time.time()

    # Convert string labels to numbers only if needed
    test_df['polarity'] = test_df['polarity'].replace({
        'positive':1,
        'negative':2,
        'neutral':0
    })

    sentences = test_df.text.values
    labels = test_df.polarity.values

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
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE
    )

    model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
    model.load_state_dict(torch.load(model_saved_path))
    model.eval()

    predictions = []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    end = time.time()
    print('Prediction used {:.2f} seconds'.format(end - begin))

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Overwrite polarity column with predicted values
    test_df['polarity'] = flat_predictions

    return test_df
