import json
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader
from hw1 import Transformer, Embedder
import torch 
import torch.nn as nn


def tokenize_data(dataset):
    seq_list = []
    start_token = "[START]"
    answer_token = "[ANSWER]"
    
    with open(dataset, "r") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                
                # Start building the sequence
                seq = f"{start_token} <{data['fact1']}> <{data['question']['stem']}> "
                
                # Add each choice with its label
                for choice in data['question']['choices']:
                    seq += f"[{choice['label']}] <{choice['text']}> "
                
                # Add the answer
                seq += f"{answer_token} <{data['answerKey']}>"
                
                seq_list.append(seq)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}, line: {line.strip()}")
            except KeyError as e:
                print(f"Missing key in JSON data: {e}, line: {line.strip()}")
    
    return seq_list

def encode_sequences(text_sequences, tokenizer):
    tokenized_data = []
    for seq in text_sequences:
        encoded = tokenizer.encode(seq, add_special_tokens=True)
        tokenized_data.extend(encoded)
    return tokenized_data

class QADataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.token_ids) // self.seq_length
        
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        inputs = self.token_ids[start:end-1]  # All except last token
        targets = self.token_ids[start+1:end] # Shifted by one
        return torch.tensor(inputs), torch.tensor(targets)

def load_pretrained_model(pretrained_path, tokenizer, special_tokens):
    # Original vocab size without special tokens
    original_vocab_size = 50257  # GPT-2's original vocab size
    d_model = 512
    
    # Initialize model with expanded vocab size
    model = Transformer(
        vocab_size=original_vocab_size + len(special_tokens),
        d_model=d_model,
        N=6,
        heads=8,
        dropout=0.1
    )
    
    # Load pretrained weights
    pretrained_dict = torch.load(pretrained_path)
    
    # 1. Handle embedding layer
    # Get original embeddings from pretrained model
    old_weights = pretrained_dict['decoder.embed.embed.weight']
    
    # Initialize new embeddings with original weights
    model.decoder.embed.embed.weight.data[:original_vocab_size] = old_weights
    
    # Initialize new tokens (using same initialization as original)
    model.decoder.embed.embed.weight.data[original_vocab_size:].normal_(
        mean=0.0,
        std=0.02
    )
    
    # 2. Handle output layer
    # Get original output weights from pretrained model
    old_output_weight = pretrained_dict['decoder.output.weight']
    old_output_bias = pretrained_dict['decoder.output.bias']
    
    # Initialize new output layer with original weights
    model.decoder.output.weight.data[:original_vocab_size] = old_output_weight
    model.decoder.output.bias.data[:original_vocab_size] = old_output_bias
    
    # Initialize new output weights (you might want different initialization here)
    model.decoder.output.weight.data[original_vocab_size:].normal_(
        mean=0.0,
        std=0.02
    )
    model.decoder.output.bias.data[original_vocab_size:].zero_()
    
    return model

def main():
    seq_length = 512
    batch_size = 8
    num_epochs = 3
    model_path = "model_weights_wiki103/model_epoch_20.pth"

    # Initialize tokenizer and model
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = ["[START]", "[A]", "[B]", "[C]", "[D]", "[ANSWER]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Prepare data
    text_sequences = tokenize_data("datasets/train_complete.jsonl")
    tokenized_ids = encode_sequences(text_sequences, tokenizer)
    dataset = QADataset(tokenized_ids, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_pretrained_model(model_path, tokenizer, special_tokens)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def create_mask(seq):
        """Create nopeak mask for sequences"""
        mask = (seq != tokenizer.pad_token_id).unsqueeze(-2)
        seq_len = seq.size(1)
        np_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool()
        if seq.is_cuda:
            np_mask = np_mask.to(device)
        return mask & ~np_mask

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Create mask
            trg_mask = create_mask(inputs)
            
            optimizer.zero_grad()
            
            # Forward pass with mask
            outputs = model(inputs, trg_mask=trg_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), 
                           targets.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader):.4f}")


main()
