import json
from transformers import GPT2TokenizerFast

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

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Add your special tokens if they don't exist
    special_tokens = ["[START]", "[A]", "[B]", "[C]", "[D]", "[ANSWER]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    text_sequences = tokenize_data("datasets/train_complete.jsonl")

    tokenized_ids = encode_sequences(text_sequences, tokenizer)

    for i in tokenized_ids:
        print(i)

main()

