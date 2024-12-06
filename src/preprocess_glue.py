import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indices = torch.arange(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.indices[idx]
    
def preprocess_function(
        examples, 
        sentence1_key,
        sentence2_key,
        tokenizer,
        padding: str,
        truncation: bool
    ):
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    return tokenizer(*args, padding=padding, truncation=truncation)

def get_glue_data(dataset_name):
    dataset = load_dataset("nyu-mll/glue", dataset_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]
    
    return train_dataset, test_dataset

def get_glue_datasets(
        dataset_name,
        sentence1_key,
        sentence2_key,
        tokenizer,
        padding: str,
        truncation: bool
    ):
    train_dataset, test_dataset = get_glue_data(dataset_name)

    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, sentence1_key, sentence2_key, tokenizer, padding, truncation), batched=True)
    test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, sentence1_key, sentence2_key, tokenizer, padding, truncation), batched=True)
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')

    return train_dataset, test_dataset

def dataset_to_dataloader(
        dataset, 
        batch_size: int, 
        shuffle: bool
    ):
    return DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=shuffle)