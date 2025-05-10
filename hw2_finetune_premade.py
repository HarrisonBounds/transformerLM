import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn
import json
import torch.optim as optim

# Define the dataset class to handle the JSONL data format
class MultipleChoiceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        stem = item["question"]["stem"]
        choices = item["question"]["choices"]
        fact = item["fact1"]
        answer_key = item["answerKey"]
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_choice_index = label_map[answer_key]

        encoded_inputs = []
        for choice in choices:
            text = f"[CLS] {fact} {stem} {choice['text']} [SEP]"
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encoded_inputs.append(encoded)

        return {
            "input_ids": torch.stack([enc["input_ids"].squeeze(0) for enc in encoded_inputs]),
            "attention_mask": torch.stack([enc["attention_mask"].squeeze(0) for enc in encoded_inputs]),
            "labels": torch.tensor(correct_choice_index)
        }

# Define the model architecture
class BertForMultipleChoiceCustom(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        batch_size, num_choices, seq_length = input_ids.size()

        input_ids_flat = input_ids.view(-1, seq_length)
        attention_mask_flat = attention_mask.view(-1, seq_length)

        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)
        reshaped_logits = logits.view(batch_size, num_choices)
        return reshaped_logits

# Function to load JSONL data
def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    # Hyperparameters (adjust as needed)
    model_name = "bert-base-uncased"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load the datasets (replace with your actual file paths)
    train_data_path = "datasets/train_complete.jsonl"
    val_data_path = "datasets/dev_complete.jsonl"
    test_data_path = "datasets/test_complete.jsonl"

    try:
        train_data = load_jsonl_data(train_data_path)
        val_data = load_jsonl_data(val_data_path)
        test_data = load_jsonl_data(test_data_path)
    except FileNotFoundError as e:
        print(f"Error: One or more data files not found: {e}")
        exit()

    # Create datasets and dataloaders
    train_dataset = MultipleChoiceDataset(train_data, tokenizer, max_length)
    val_dataset = MultipleChoiceDataset(val_data, tokenizer, max_length)
    test_dataset = MultipleChoiceDataset(test_data, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    model = BertForMultipleChoiceCustom(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        # Evaluation on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")

    # Evaluation on test set
    print("\nStarting evaluation on the test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            test_total += labels.size(0)
            test_correct += (predictions == labels).sum().item()

    test_accuracy = test_correct / test_total
    print(f"Fine-tuned Test Accuracy: {test_accuracy:.4f}")

    # Zero-shot evaluation (evaluate the pre-trained model)
    print("\nPerforming zero-shot evaluation on validation and test sets...")
    pretrained_model = BertForMultipleChoiceCustom(model_name).to(device)
    pretrained_model.eval()

    zero_shot_val_correct = 0
    zero_shot_val_total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = pretrained_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            zero_shot_val_total += labels.size(0)
            zero_shot_val_correct += (predictions == labels).sum().item()
    zero_shot_val_accuracy = zero_shot_val_correct / zero_shot_val_total
    print(f"Zero-Shot Validation Accuracy: {zero_shot_val_accuracy:.4f}")

    zero_shot_test_correct = 0
    zero_shot_test_total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = pretrained_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            zero_shot_test_total += labels.size(0)
            zero_shot_test_correct += (predictions == labels).sum().item()
    zero_shot_test_accuracy = zero_shot_test_correct / zero_shot_test_total
    print(f"Zero-Shot Test Accuracy: {zero_shot_test_accuracy:.4f}")
