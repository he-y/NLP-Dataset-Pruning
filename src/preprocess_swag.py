import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch

class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indices = torch.arange(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.indices[idx]

def preprocess_function(examples, tokenizer):
    ending_names = ["ending0", "ending1", "ending2", "ending3"]

    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=128, padding="max_length")

    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def get_swag_datasets(tokenizer):
    # Load the SWAG dataset
    swag = load_dataset("swag", "regular")

    tokenized_swag = swag.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    train_dataset = tokenized_swag["train"]
    validation_dataset = tokenized_swag["validation"]

    train_dataset.set_format("torch")
    validation_dataset.set_format("torch")

    return train_dataset, validation_dataset

def dataset_to_dataloader(
        dataset, 
        batch_size: int, 
        shuffle: bool
    ):
    return DataLoader(
        IndexedDataset(dataset), 
        batch_size=batch_size, 
        shuffle=shuffle
    )