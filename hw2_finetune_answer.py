import json
from torch.utils.data import Dataset, DataLoader
from hw1 import Transformer, Embedder
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import GPT2TokenizerFast
from rouge import Rouge
from bert_score import score

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

                # Add the actual text answer
                seq += f"{answer_token} <{data.get('answerText', 'N/A')}>"  # Assuming 'answerText' field

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
        inputs = self.token_ids[start:end - 1]  # All except last token
        targets = self.token_ids[start + 1:end]  # Shifted by one
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
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))

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


def create_mask(seq, tokenizer, device):
    """Create nopeak mask for sequences"""
    mask = (seq != tokenizer.pad_token_id).unsqueeze(-2)
    seq_len = seq.size(1)
    np_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool()
    if seq.is_cuda:
        np_mask = np_mask.to(device)
    return mask & ~np_mask


def focused_loss(outputs, targets, inputs, answer_token_id, criterion, tokenizer):
    # Standard language modeling loss
    lm_loss = criterion(outputs.view(-1, outputs.size(-1)),
                       targets.view(-1))

    # Find the starting position of the answer
    answer_start_mask = (inputs[:, :-1] == answer_token_id)

    if answer_start_mask.any():
        # Get the indices where [ANSWER] token appears (excluding the last token)
        answer_start_indices = torch.where(answer_start_mask)

        # Extract the relevant outputs and targets for the answer
        answer_losses = []
        for batch_index, start_index in zip(*answer_start_indices):
            # Consider the sequence of tokens following [ANSWER] as the target answer
            # You might need to define a maximum answer length to consider
            max_answer_length = 20  # Adjust as needed
            relevant_output = outputs[batch_index, start_index + 1:start_index + 1 + max_answer_length]
            relevant_target = targets[batch_index, start_index + 2:start_index + 2 + max_answer_length]  # +2 because target is shifted

            # Ensure both have the same length for loss calculation
            valid_length = min(relevant_output.size(0), relevant_target.size(0))
            if valid_length > 0:
                answer_losses.append(criterion(relevant_output[:valid_length], relevant_target[:valid_length]))

        if answer_losses:
            avg_answer_loss = torch.mean(torch.stack(answer_losses))
            return 0.2 * lm_loss + 0.8 * avg_answer_loss
    return lm_loss


def evaluate_model(model, dataset, tokenizer, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    answer_token_id = tokenizer.encode("[ANSWER]")[0]
    start_token_id = tokenizer.encode("[START]")[0]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    total_bleu_score = 0.0
    total_rouge_score = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    predictions = []
    references = []
    num_examples = 0
    smoothing = SmoothingFunction().method1
    rouge = Rouge()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            trg_mask = create_mask(inputs, tokenizer, device)
            outputs = model(inputs, trg_mask=trg_mask)

            for i in range(inputs.size(0)):
                answer_start_indices = (inputs[i] == answer_token_id).nonzero(as_tuple=True)[0]

                if len(answer_start_indices) > 0:
                    answer_start_index = answer_start_indices[0] + 1

                    generated_tokens = []
                    current_input = inputs[i].unsqueeze(0).to(device)
                    max_generation_length = 30

                    for _ in range(max_generation_length):
                        prediction = model(current_input, trg_mask=create_mask(current_input, tokenizer, device))[:, -1, :]
                        predicted_token_id = torch.argmax(prediction, dim=-1).item()

                        if predicted_token_id == eos_token_id or predicted_token_id == pad_token_id:
                            break
                        generated_tokens.append(predicted_token_id)
                        current_input = torch.cat((current_input, torch.tensor([[predicted_token_id]]).to(device)), dim=1)

                    predicted_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    target_answer_start_index = (targets[i] == answer_token_id).nonzero(as_tuple=True)[0]
                    ground_truth_answer = ""
                    if len(target_answer_start_index) > 0:
                        target_answer_start = target_answer_start_index[0] + 1
                        target_answer_tokens = []
                        for j in range(target_answer_start, targets.size(1)):
                            if targets[i, j] == eos_token_id or targets[i, j] == pad_token_id or len(
                                    target_answer_tokens) >= max_generation_length:
                                break
                            target_answer_tokens.append(targets[i, j].item())
                        ground_truth_answer = tokenizer.decode(target_answer_tokens, skip_special_tokens=True).strip()

                    if ground_truth_answer:
                        reference = ground_truth_answer.split()
                        candidate = predicted_answer.split()
                        bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
                        total_bleu_score += bleu

                        rouge_scores_list = rouge.get_scores(predicted_answer, ground_truth_answer)
                        if rouge_scores_list:
                            rouge_scores = rouge_scores_list[0]
                            total_rouge_score["rouge1"] += rouge_scores.get('rouge-1', {}).get('f', 0)
                            total_rouge_score["rouge2"] += rouge_scores.get('rouge-2', {}).get('f', 0)
                            total_rouge_score["rougeL"] += rouge_scores.get('rouge-l', {}).get('f', 0)

                        predictions.append(predicted_answer)
                        references.append(ground_truth_answer)
                        num_examples += 1

    avg_bleu_score = total_bleu_score / max(1, num_examples)
    avg_rouge_score = {key: value / max(1, num_examples) for key, value in total_rouge_score.items()}

    P, R, F1 = score(predictions, references, lang='en', verbose=True, device=device)
    avg_bertscore_f1 = F1.mean().item()

    print(f"\nEvaluation Results over {num_examples} examples:")
    print(f"Average BLEU score: {avg_bleu_score:.4f}")
    print(f"Average ROUGE scores: {avg_rouge_score}")
    print(f"Average BERTscore F1: {avg_bertscore_f1:.4f}")

    return avg_bleu_score, avg_rouge_score, avg_bertscore_f1


def verify_tokenization(tokenizer):
    print("\nToken Verification:")
    for label in ["[A]", "[B]", "[C]", "[D]"]:
        token_id = tokenizer.encode(label)[0]
        print(f"{label} -> ID: {token_id} -> Decoded: {tokenizer.decode([token_id])}")

    answer_token = "[ANSWER]"
    answer_id = tokenizer.encode(answer_token)[0]
    print(f"{answer_token} -> ID: {answer_id} -> Decoded: {tokenizer.decode([answer_id])}")


def main():
    seq_length = 512
    batch_size = 16
    num_epochs = 20
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    verify_tokenization(tokenizer)

    answer_token_id = tokenizer.encode("[ANSWER]")[0]
    choice_tokens = {
        tokenizer.encode("[A]")[0]: "A",
        tokenizer.encode("[B]")[0]: "B",
        tokenizer.encode("[C]")[0]: "C",
        tokenizer.encode("[D]")[0]: "D"
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            trg_mask = create_mask(inputs, tokenizer, device)

            optimizer.zero_grad()
            outputs = model(inputs, trg_mask=trg_mask)

            loss = focused_loss(outputs, targets, inputs, answer_token_id, criterion, tokenizer)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

    # Validation and Testing will use the modified evaluate_model
    val_sequences = tokenize_data("datasets/dev_complete.jsonl")
    test_sequences = tokenize_data("datasets/test_complete.jsonl")

    val_tokenized = encode_sequences(val_sequences, tokenizer)
    test_tokenized = encode_sequences(test_sequences, tokenizer)

    val_dataset = QADataset(val_tokenized, seq_length)
    test_dataset = QADataset(test_tokenized, seq_length)

    val_bleu, val_rouge, val_bertscore = evaluate_model(model, val_dataset, tokenizer, device)
    test_bleu, test_rouge, test_bertscore = evaluate_model(model, test_dataset, tokenizer, device)

    print(f"\nValidation BLEU: {val_bleu:.4f}")
    print(f"Validation ROUGE: {val_rouge}")
    print(f"Validation BERTscore F1: {val_bertscore:.4f}")
    print(f"\nTest BLEU: {test_bleu:.4f}")
    print(f"Test ROUGE: {test_rouge}")
    print(f"Test BERTscore F1: {test_bertscore:.4f}")

    torch.save(model.state_dict(), "finetuned_model_text_answer.pth")


if __name__ == "__main__":
    main()