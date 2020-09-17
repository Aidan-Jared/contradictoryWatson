import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

class Embeder(nn.Module):
    # find the meaning of the word
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    # the words position
    def __init__(self, d_model, max_seq_len = 300):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1) / d_model))))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len].requires_grad_(False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def _attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask = None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch size * h * seq len * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = self._attention(q, k , v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        
        output = self.out(concat)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads,d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x2 = self.attn(x2,x2,x2, mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x2 = self.ff(x2)
        x = x + self.dropout_2(x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)

        self.ff = FeedForward(d_model)

    def forward(self, x, e_ouputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x2 = self.attn_1(x2,x2,x2, trg_mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x2 = self.attn_2(x2, e_ouputs, e_ouputs, src_mask)
        x = x + self.dropout_2(x2)
        x2 = self.norm_3(x)
        x2 = self.ff(x2)
        x = x + self.dropout_3(x2)
        return x

def get_clones(modlue, N):
    return nn.ModuleList([copy.deepcopy(modlue) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embeder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embeder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_ouputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_ouputs, src_mask, trg_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, output_dim , d_model, N, heads, input_pad):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(src_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, output_dim)
        self.norm = Norm(output_dim)
        self.input_pad = input_pad
    
    def forward(self, src, trg):
        prm_mask = self._src_mask(src)
        hyp_mask = self._src_mask(trg)
        e_ouputs = self.encoder(src, prm_mask)
        d_output = self.decoder(trg, e_ouputs, prm_mask, hyp_mask)
        cls_output = d_output[:,0]
        output = self.out(cls_output)
        return output

    def _src_mask(self, batch):
        input_mask = (batch != self.input_pad).unsqueeze(-2)
        return input_mask

        size = batch.size(1)
        m = np.ones((1, size, size))
        nopeak_mask = np.triu(m,k=1).astype('uint8')
        nopeak_mask = torch.from_numpy(nopeak_mask) == 0

        target_mask = target_mask & nopeak_mask
        # target_mask = F.pad(target_mask, pad=(0, 0,target_seq.size(0)-size,0), value=False)
        return target_mask