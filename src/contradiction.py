import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.data import Field
from Transformer import Transformer

import numpy as np
import pandas as pd
import math
import time
import spacy

nlp = spacy.load("xx_ent_wiki_sm")

def tokenizer(doc):
    tokens = [x.text for x in nlp.tokenizer(doc) if not x.is_space]
    return tokens

def train_model(model, iterator, val_iter, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    res = []
    targets = []
    for iteration ,batch in enumerate(iterator):
        prem = batch.prem.transpose(0,1)
        hyp = batch.hyp.transpose(0,1)
        lang = batch.lang_a
        trg = batch.label

        inputs = torch.cat((prem, hyp[:,1:]),1)
        
        optimizer.zero_grad()
        output = model(inputs, lang)

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        
        output = F.softmax(output, dim=1)
        values, ix = output.data.topk(1)
        for index, i in enumerate(ix):
            res.append(i)
            targets.append(trg[index])
    epoch_acc = accuracy(torch.tensor(res), torch.tensor(targets))
    val_acc = evaluate(model, val_iter) 
    return epoch_loss / len(iterator), epoch_acc, val_acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def accuracy(pred, label):
    correct = (pred == label).sum().item()
    acc = (correct / pred.size(0)) * 100
    return acc

def evaluate(model, example):
    model.eval()
    res = []
    targets = []
    with torch.no_grad():
        for iteration ,batch in enumerate(example):
            prem = batch.prem.transpose(0,1)
            hyp = batch.hyp.transpose(0,1)
            lang = batch.lang_a
            trg = batch.label

            inputs = torch.cat((prem, hyp[:,1:]),1)

            output = model(inputs, lang)
            output = F.softmax(output, dim=1)
            val, ix = output.data.topk(1)
            for index, i in enumerate(ix):
                res.append(i)
                targets.append(trg[index])
    acc = accuracy(torch.tensor(res), torch.tensor(targets))
    return acc

if __name__ == "__main__":
    
    TEXT = Field(
                tokenize= tokenizer,
                init_token= '<cls>',
                eos_token= '<sep>')
    
    CAT = Field(sequential=False)
    TRG = Field(sequential=False, is_target=True, unk_token=None)
    fields = [('id', CAT), ('prem', TEXT), ('hyp', TEXT), ('lang_a', CAT), ('lang', CAT), ('label', TRG)]

    train_data, test_data = torchtext.data.TabularDataset.splits(
                                                    path='data/',
                                                    train='train.csv',
                                                    test='test.csv',
                                                    format='csv',
                                                    skip_header=True,
                                                    fields=fields
                                                )
    
    train_data, val_data = train_data.split(split_ratio=.7)

    TEXT.build_vocab(train_data)
    TRG.build_vocab(train_data)
    CAT.build_vocab(train_data.lang_a)

    BATCH_SIZE = 80
    
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
                                                                (train_data, val_data, test_data),
                                                                batch_size=BATCH_SIZE,
                                                                sort=False
                                                            )

    INPUT_DIM = len(TEXT.vocab)
    NUM_LANG = len(CAT.vocab)
    OUTPUT_DIM = 3
    d_model = 512
    heads = 8
    N = 5
    PAD_IDX = TEXT.vocab.stoi['<pad>']

    model = Transformer(INPUT_DIM, NUM_LANG, OUTPUT_DIM, d_model, N, heads, PAD_IDX)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = optim.AdamW(model.parameters(), lr=.001)

    criterion = nn.CrossEntropyLoss()

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc, val_acc = train_model(model, train_iter, val_iter, optimizer, criterion, CLIP)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tTrain acc: {train_acc:.3f}| Val acc: {val_acc:7.3f}')