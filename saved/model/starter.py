import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast
import re 
import matplotlib.pyplot as plt

def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def euclidean_distance(q, k):
    """Calculates the Euclidean distance between query and key."""
    return torch.sqrt(torch.sum((q.unsqueeze(-2) - k.unsqueeze(-3)) ** 2, dim=-1))

def distance_based_attention(q, k, v, d_k, mask=None, dropout=None):
    """Attention based on Euclidean distance."""
    distances = euclidean_distance(q, k)  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

    # Convert distances to scores. Smaller distance means higher score.
    # You might need to experiment with the scaling factor (-1/sqrt(d_k) is common)
    scores = -distances / math.sqrt(d_k) # Invert and scale

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, v)
    return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = distance_based_attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        #self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        #self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        #self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        # src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        x = self.norm(x)
        x = self.output(x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        #self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(vocab_size, d_model, N, heads, dropout)
        #self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, trg, trg_mask):
        #e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, trg_mask)
        #output = self.out(d_output)
        return d_output

def get_model(opt):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(opt.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model

def nopeak_mask(size):
    """
    Creates a nopeak mask to prevent the model from attending to future positions.
    Args:
        size (int): The length of the sequence.
    Returns:
        torch.Tensor: A tensor of shape (1, size, size) where the lower triangular
                      part is filled with 1 and the upper triangular part is filled
                      with 0.
    """
    np_mask = np.triu(np.ones((1, size, size)),
                       k=1).astype('uint8')
    torch_mask = torch.from_numpy(np_mask) == 0
    return torch_mask

def data_generator(data, batch_size, seq_len, device, tokenizer):
    """Generates batches of data with padding.

    Args:
        data (list):  List of token IDs (output of read_corpus).
        batch_size (int): The desired batch size.
        seq_len (int):  Maximum sequence length.
        tokenizer:   The tokenizer.

    Yields:
        torch.Tensor:  Padded batch of input sequences (shape: batch_size, seq_len).
        torch.Tensor:  Padded batch of target sequences (shape: batch_size, seq_len).
    """
    for i in range(0, len(data) - seq_len, seq_len * batch_size): #modified the loop
        batch_data = data[i:i + seq_len * batch_size]
        
        # Create input and target sequences
        inputs = []
        targets = []
        for j in range(0, len(batch_data), seq_len):
            inputs.append(batch_data[j:j + seq_len - 1])
            targets.append(batch_data[j + 1:j + seq_len])

        # Pad sequences to seq_len
        padded_inputs = []
        padded_targets = []

        for seq in inputs:
            padding_len = seq_len - len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * padding_len
            padded_inputs.append(padded_seq)

        for seq in targets:
            padding_len = seq_len - len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * padding_len
            padded_targets.append(padded_seq)
            
        input_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(device) #global opt
        target_tensor = torch.tensor(padded_targets, dtype=torch.long).to(device)
        
        yield input_tensor, target_tensor

    
def train_model(model, opt, tokenizer):

    model.train()
    optimizer = opt.optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(opt.epochs):
        total_loss = 0
        num_batches = 0
        total_tokens_processed = 0
        start_time_epoch = time.time()

        for i, (input_batch, target_batch) in enumerate(data_generator(opt.train, opt.batchsize, opt.seqlen, opt.device, tokenizer)):
            start_time_batch = time.time()
            optimizer.zero_grad()

            batch_size, seq_len = input_batch.size()
            trg_mask = nopeak_mask(seq_len).to(opt.device)

            output = model(input_batch, trg_mask)
            output = output.reshape(batch_size * seq_len, -1)
            target_batch = target_batch.reshape(batch_size * seq_len)

            loss = criterion(output, target_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.norm)
            optimizer.step()
            if opt.SGDR:
                opt.sched.step()

            total_loss += loss.item()
            num_batches += 1

            # Calculate tokens processed in this batch
            tokens_in_batch = input_batch.numel() + target_batch.numel()
            total_tokens_processed += tokens_in_batch

            end_time_batch = time.time()
            batch_time = end_time_batch - start_time_batch
            tokens_per_second_batch = tokens_in_batch / batch_time if batch_time > 0 else 0

        # Calculate average loss and perplexity for the epoch
        avg_loss_epoch = total_loss / num_batches if num_batches > 0 else 0.
        perplexity_epoch = torch.exp(torch.tensor(avg_loss_epoch)).item() if avg_loss_epoch > 0 else 0.

        # Calculate tokens per second for the entire epoch
        end_time_epoch = time.time()
        epoch_time = end_time_epoch - start_time_epoch
        tokens_per_second_epoch = total_tokens_processed / epoch_time if epoch_time > 0 else 0

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss_epoch:.4f}, Train Perplexity: {perplexity_epoch:.2f}, Tokens/sec (epoch): {tokens_per_second_epoch:.2f}")
        with open(opt.log_file, "a") as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {avg_loss_epoch:.4f}, Train Perplexity: {perplexity_epoch:.2f}\n")

        # Save model
        if opt.savename:
            torch.save(model.state_dict(), f"{opt.savename}/model_epoch_{epoch+1}.pth")

        validate_model(model, opt, epoch, tokenizer)
        test_model(model, opt, epoch, tokenizer)
    
def validate_model(model, opt, epoch, tokenizer):

    model.eval()
    total_loss = 0
    num_batches = 0  # Initialize batch counter for validation set
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_generator(opt.valid, opt.batchsize, opt.seqlen, opt.device, tokenizer)):  # Use data_generator for validation data
            batch_size, seq_len = input_batch.size()
            trg_mask = nopeak_mask(seq_len).to(opt.device)
            output = model(input_batch, trg_mask)
             # Output is of shape (batch_size, seq_len, vocab_size)
            output = output.reshape(batch_size * seq_len, -1)
            target_batch = target_batch.reshape(batch_size * seq_len)
            loss = criterion(output, target_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else 0.
    print(f"Epoch {epoch+1}, Validation Loss: {avg_loss:.4f}, Validation Perplexity: {perplexity:.2f}")
    with open(opt.log_file, "a") as f:
        f.write(f"Epoch {epoch+1}, Validation Loss: {avg_loss:.4f}, Validation Perplexity: {perplexity:.2f}\n")
    

def test_model(model, opt, epoch, tokenizer):
    
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_generator(opt.test, opt.batchsize, opt.seqlen, opt.device, tokenizer)):  # Use data_generator
            batch_size, seq_len = input_batch.size()
            trg_mask = nopeak_mask(seq_len).to(opt.device)
            output = model(input_batch, trg_mask)
             # Output is of shape (batch_size, seq_len, vocab_size)
            output = output.reshape(batch_size * seq_len, -1)
            target_batch = target_batch.reshape(batch_size * seq_len)
            loss = criterion(output, target_batch)
            total_loss += loss.item()

    avg_loss = total_loss / (i + 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Epoch {epoch+1}, Test Loss: {avg_loss:.4f}, Test Perplexity: {perplexity:.2f}")
    with open(opt.log_file, "a") as f:
        f.write(f"Epoch {epoch+1}, Test Loss: {avg_loss:.4f}, Test Perplexity: {perplexity:.2f}\n")
    model.train()


def plot_metrics(log_file):
    metrics = {}

    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if epoch not in metrics:
                    metrics[epoch] = {}

            train_loss_match = re.search(r'Train Loss: (\d+\.\d+)', line)
            if train_loss_match and epoch in metrics:
                metrics[epoch]['train_loss'] = float(train_loss_match.group(1))

            train_perplexity_match = re.search(r'Train Perplexity: (\d+\.\d+)', line)
            if train_perplexity_match and epoch in metrics:
                metrics[epoch]['train_perplexity'] = float(train_perplexity_match.group(1))

            val_loss_match = re.search(r'Validation Loss: (\d+\.\d+)', line)
            if val_loss_match and epoch in metrics:
                metrics[epoch]['val_loss'] = float(val_loss_match.group(1))

            val_perplexity_match = re.search(r'Validation Perplexity: (\d+\.\d+)', line)
            if val_perplexity_match and epoch in metrics:
                metrics[epoch]['val_perplexity'] = float(val_perplexity_match.group(1))

            test_loss_match = re.search(r'Test Loss: (\d+\.\d+)', line)
            if test_loss_match and epoch in metrics:
                metrics[epoch]['test_loss'] = float(test_loss_match.group(1))

            test_perplexity_match = re.search(r'Test Perplexity: (\d+\.\d+)', line)
            if test_perplexity_match and epoch in metrics:
                metrics[epoch]['test_perplexity'] = float(test_perplexity_match.group(1))

    epochs = sorted(metrics.keys())
    train_losses = [metrics[e].get('train_loss') for e in epochs]
    train_perplexities = [metrics[e].get('train_perplexity') for e in epochs]
    val_losses = [metrics[e].get('val_loss') for e in epochs]
    val_perplexities = [metrics[e].get('val_perplexity') for e in epochs]
    test_losses = [metrics[e].get('test_loss') for e in epochs]
    test_perplexities = [metrics[e].get('test_perplexity') for e in epochs]

    # Plotting Losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', linestyle='-', label='Validation Loss')
    plt.plot(epochs, test_losses, marker='o', linestyle='-', label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_from_log_distance.png')
    plt.show()

    # Plotting Perplexities
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_perplexities, marker='o', linestyle='-', label='Train Perplexity')
    plt.plot(epochs, val_perplexities, marker='o', linestyle='-', label='Validation Perplexity')
    plt.plot(epochs, test_perplexities, marker='o', linestyle='-', label='Test Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training, Validation, and Test Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig('perplexity_plot_from_log_distance.png')
    plt.show()

def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=8)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str, default='model_weights_distance')    
    parser.add_argument('-loadname', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "/"
    opt.dir_name = dir_name
    shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file_distance.txt"
    
    print(str(opt))
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    opt.train = read_corpus('wiki2.train.txt',tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('wiki2.test.txt',tokenizer)
    
    # opt.vocab_size = len(tokenizer)
    # obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()
    
    model = get_model(opt)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0
            
    train_model(model, opt, tokenizer)
    plot_metrics(opt.log_file)
    
        
if __name__ == "__main__":
    main()        