import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field

import numpy as np
import pandas as pd

if __name__ == "__main__":
    TEXT = Field(
                tokenize= 'spacy',
                tokenizer_language= "xx_ent_wiki_sm",
                init_token= '<sos>',
                eos_token= '<eos>',
                lower= True)
    
    TRG = Field(sequential=False)
    fields = [('id', TRG), ('prem', TEXT), ('hyp', TEXT), ('lang_a', TRG), ('lang', TRG), ('label', TRG)]

    train, test = torchtext.data.TabularDataset.splits(
                                                    path='data/',
                                                    train='train.csv',
                                                    test='test.csv',
                                                    format='csv',
                                                    skip_header=True,
                                                    fields=fields
                                                )

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
                                                                (train, test),
                                                                batch_sizes=(16,256)
                                                            )

    TEXT.build_vocab(train)
    print('here')