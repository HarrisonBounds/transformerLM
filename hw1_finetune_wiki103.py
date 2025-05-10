import json

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

# Example usage
tokenized_data = tokenize_data("datasets/train_complete.jsonl")

# Print the first few examples to verify
for i, seq in enumerate(tokenized_data[:3]):
    print(f"Example {i+1}:")
    print(seq)
    print()