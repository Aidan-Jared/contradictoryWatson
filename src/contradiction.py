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

def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _,batch in enumerate(iterator):
        prem = batch.prem.transpose(0,1)
        hyp = batch.hyp.transpose(0,1)
        trg = batch.label
        
        optimizer.zero_grad()
        output = model(prem, hyp)

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate(model, iterator):
    model.eval()
    res = []
    with torch.no_grad():
        for _,batch in enumerate(iterator):
            prem = batch.prem.transpose(0,1)
            hyp = batch.hyp.transpose(0,1)

            output = model(prem, hyp)
            output = F.softmax(output)
            val, ix = output[:-1].data.topk(1)
            res.append(TRG.vocab.stoi[ix])
    return res

if __name__ == "__main__":
    
    TEXT = Field(
                tokenize= tokenizer,
                init_token= '<cls>',
                eos_token= '<sep>')
    
    TRG = Field(sequential=False)
    fields = [('id', TRG), ('prem', TEXT), ('hyp', TEXT), ('lang_a', TEXT), ('lang', TRG), ('label', TRG)]

    train_data, test_data = torchtext.data.TabularDataset.splits(
                                                    path='data/',
                                                    train='train.csv',
                                                    test='test.csv',
                                                    format='csv',
                                                    skip_header=True,
                                                    fields=fields
                                                )

    TEXT.build_vocab(train_data)
    TRG.build_vocab(train_data.label)
    
    train_iter, test_iter = torchtext.data.BucketIterator.splits(
                                                                (train_data, test_data),
                                                                batch_sizes=(16,256)
                                                            )

    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    d_model = 512
    heads = 8
    N = 1
    PAD_IDX = TEXT.vocab.stoi['<pad>']

    model = Transformer(INPUT_DIM, OUTPUT_DIM, d_model, N, heads, PAD_IDX)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = optim.Adam(model.parameters(), lr=.0001, betas=(.9,.98), eps=1e-9)

    criterion = nn.CrossEntropyLoss()

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_model(model, train_iter, optimizer, criterion, CLIP)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    results = evaluate(model, train)