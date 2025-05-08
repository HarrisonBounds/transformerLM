from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np
from torch.utils.data import Dataset

class MCQADataset(Dataset):
    def __init__(self, encoded_data, labels):
        self.encoded_data = encoded_data
        self.labels = labels  

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        encoded_choices = self.encoded_data[idx] 
        correct_answer_index = self.labels[idx]

        input_ids = [item[0].squeeze(0) for item in encoded_choices] # Remove batch dimension
        attention_mask = [item[1].squeeze(0) for item in encoded_choices] 

        return {
            'input_ids': torch.stack(input_ids),      
            'attention_mask': torch.stack(attention_mask), 
            'labels': torch.tensor(correct_answer_index) 
        }

def encode_question_choices_with_label(choices, tokenizer, max_length=128):
    encoded_inputs = []
    correct_answer_index = -1
    for i, (text, label) in enumerate(choices):
        encoded = tokenizer.encode_plus(
            "[CLS] " + text.replace(" [SEP]", " [END]"),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoded_inputs.append((encoded['input_ids'], encoded['attention_mask']))
        if label == 1:
            correct_answer_index = i
    return encoded_inputs, correct_answer_index


def main():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []
    
    dataset_path = "datasets/"
    file_name = dataset_path + 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        
        print(obs)
        print(' ')
        
        print(result['question']['stem'])
        print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        print('  Fact: ',result['fact1'])
        print('  Answer: ',result['answerKey'])
        print('  ')
                
    file_name = dataset_path + 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        valid.append(obs)
        
    file_name = dataset_path + 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.nn.Linear(768,4)
    softmax = torch.nn.Softmax(dim=1)
    
#    Add code to fine-tune and test your MCQA classifier.

    # 1. Encode data
    train_encoded = []
    valid_encoded = []
    test_encoded = []

    train_labels = []
    valid_labels = []
    test_labels = []

    for obs in train:
        encoded, label = encode_question_choices_with_label(obs, tokenizer)
        train_encoded.append(encoded)
        train_labels.append(label)
       
    for obs in train:
        encoded, label = encode_question_choices_with_label(obs, tokenizer)
        train_encoded.append(encoded)
        train_labels.append(label)

    for obs in train:
        encoded, label = encode_question_choices_with_label(obs, tokenizer)
        train_encoded.append(encoded)
        train_labels.append(label)

    train_dataset = MCQADataset(train_encoded, train_labels)
    valid_dataset = MCQADataset(valid_encoded, valid_labels)
    test_dataset = MCQADataset(test_encoded, test_labels)

    model.eval()
    with torch.no_grad():
        for question in train_encoded:
            for input_ids, attention_mask in question:
                outputs = model(input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                logits = linear(cls_embedding)
                probabilities = softmax(logits)
                #print("Probabilities for a choice:", probabilities)



if __name__ == "__main__":
    main()