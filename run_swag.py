import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

from src.model import get_bert_mc_model
from src.preprocess_swag import *

parser = argparse.ArgumentParser(description='training SWAG')

parser.add_argument(
    '--cuda_device',
    type=int,
    nargs='+',
    default=[0],
)

parser.add_argument(
    '--model_path',
    type=str,
    default='distilbert-base-uncased',
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    metavar='N',
)

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
)

parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
)

parser.add_argument(
    '--lr',
    type=float,
    default=1e-5,
    metavar='LR',
)

parser.add_argument(
    '--sorting_file',
    default="none",
)

parser.add_argument(
    '--get_stats',
    action='store_true',
)

parser.add_argument(
    '--input_dir',
)

parser.add_argument(
    '--output_dir', required=True, help='directory where to save results')

# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'seed', 'epochs', 'lr', 'batch_size', 'sorting_file'
]

# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args_dict = vars(args)
save_fname = '__'.join('{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set logging configuration
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'logs/swag/{save_fname}.log', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create a FileHandler that opens the file in 'w' mode (write, truncate existing)
if not os.path.exists(f'logs/swag'):
    os.makedirs(f'logs/swag')

file_handler = logging.FileHandler(f'logs/swag/{save_fname}.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the FileHandler to the root logger
logging.root.addHandler(file_handler)
logging.root.setLevel(logging.INFO)

# Redirect stdout to the Logger instance
sys.stdout = Logger()

# Get current date and time
current_datetime = datetime.now()

print('Save file path:', save_fname)

# Print the arguments
print("\nParameter configuration:")
for k, v in args_dict.items():
    if k in ordered_args:
        print(f"{k}: {v}")

# Set random seed for initialization
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

#-----------------------------------------------------------------------------------------

# Train model for 1 epoch
def train(
        args, 
        model, 
        device, 
        train_dataloader, 
        optimizer,
        epoch, 
        example_stats
    ):
    train_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    model.train()

    for batch_idx, (batch, batch_indices) in enumerate(train_dataloader):
        # Map to device
        input_ids = batch["input_ids"].to(device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        label = batch["label"]
        label_n_hot = F.one_hot(label, num_classes=4).to(device, dtype=torch.float)

        # Forward propagation, compute loss, get predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        outputs = nn.Softmax(dim=1)(logits)
        loss = criterion(outputs, label_n_hot)
        loss = loss.mean()
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()
        train_loss += loss.item()
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum().item()     

        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted.numpy())
        results = compute_metrics(all_labels, all_predictions)

        # Update loss, backprop, and step optimizer
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Calculate stats on training examples
        if args.get_stats:
            if batch_idx % 1200 == 0 or batch_idx == len(train_dataloader) - 1:
                example_stats = test_on_train_and_calc_score(args, model, device, train_dataloader, example_stats)
                elapsed_time = time.time() - start_time
                print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        # Print LR
        curr_lr = scheduler.get_last_lr()[0]

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Batch [%3d] LR: %.6f Loss: %.4f Acc: %.3f%%' %
            (epoch + 1, args.epochs, batch_idx + 1, curr_lr, loss.item(), 100. * correct / total))
        sys.stdout.flush()

    # Add training accuracy to dict
    results = compute_metrics(all_labels, all_predictions)
    index_stats = example_stats.get('train', [[], [], [], [], []])
    index_stats[0].append(train_loss / len(train_dataloader))
    index_stats[1].append(results['accuracy'])
    index_stats[2].append(results['f1'])
    index_stats[3].append(results['precision'])
    index_stats[4].append(results['recall'])
    example_stats['train'] = index_stats
    print(' Avg loss: ', train_loss / len(train_dataloader))

def test_on_train_and_calc_score(
        args, 
        model, 
        device, 
        train_dataloader, 
        example_stats
    ):

    model.eval()

    with torch.no_grad():
        for batch_idx, (batch, batch_indices) in enumerate(train_dataloader):
            # Map to device
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["label"]
            labels_n_hot = F.one_hot(labels, num_classes=4).to(device, dtype=torch.float)

            # Forward propagation, get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            outputs = nn.Softmax(dim=1)(logits)
            loss = criterion(outputs, labels_n_hot)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()

            # # Update statistics and loss
            acc = predicted == labels
            batch_indices = np.array(batch_indices)

            # Calculate EL2N score
            error_vector = outputs - labels_n_hot
            el2n = torch.norm(error_vector, p=2, dim=1)

            for j, index in enumerate(batch_indices):
                # Get index in original dataset (not sorted by forgetting)
                index_in_original_dataset = index

                # Compute missclassification margin
                output_correct_class = logits.data[j, labels[j].item()]  # output for correct class
                sorted_output, _ = torch.sort(logits.data[j, :])

                if acc[j]:
                    # Example classified correctly, highest incorrect class is 2nd largest output
                    output_highest_incorrect_class = sorted_output[-2]
                else:
                    # Example misclassified, highest incorrect class is max output
                    output_highest_incorrect_class = sorted_output[-1]
                margin = output_correct_class.item() - output_highest_incorrect_class.item()

                # Add the statistics of the current training example to dictionary
                index_stats = example_stats.get(index_in_original_dataset, [[], [], [], []])
                index_stats[0].append(loss[j].item())
                index_stats[1].append(acc[j].sum().item())
                index_stats[2].append(margin)
                index_stats[3].append(el2n[j].item())
                example_stats[index_in_original_dataset] = index_stats

    model.train()

    return example_stats

# Test model
def test(args, 
         model, 
         device,
         test_dataloader,
         criterion,
         epoch,
         example_stats
    ):
    """
    raw_test_dataset: original test dataset loaded from squad
    eval_set_full: tokenized raw_test_dataset, including duplicated questions
    test_dataloader: dataloader for eval_set_full
    """
    test_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (batch, batch_indices) in enumerate(test_dataloader):
            # Map to device
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            label = batch["label"]
            label_n_hot = F.one_hot(label, num_classes=4).to(device, dtype=torch.float)

            # Forward propagation, compute loss, get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            outputs = nn.Softmax(dim=1)(logits)
            loss = criterion(outputs, label_n_hot)
            loss = loss.mean()

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            test_loss += loss.item()

            total += label.size(0)
            correct += predicted.eq(label).cpu().sum().item()  

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.numpy())
        
        # Calculate metrics
        results = compute_metrics(all_labels, all_predictions)

        index_stats = example_stats.get('test', [[], [], [], [], []])
        index_stats[0].append(test_loss / len(test_dataloader))
        index_stats[1].append(results['accuracy'])
        index_stats[2].append(results['f1'])
        index_stats[3].append(results['precision'])
        index_stats[4].append(results['recall'])

        example_stats['test'] = index_stats
        print("\n| Validation Epoch #%d\tAcc: %.2f%% F1: %.2f P: %.2f R: %.2f" %
                (epoch + 1, results['accuracy'] * 100, results['f1'] * 100, results['precision'] * 100, results['recall'] * 100))
        
def compute_metrics(all_labels, all_predictions):
    accuracy = sum([1 for i in range(len(all_labels)) if all_labels[i] == all_predictions[i]]) / len(all_labels)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#-----------------------------------------------------------------------------------------

# Load the BERT model
tokenizer, model = get_bert_mc_model(args.model_path)

device_args = args.cuda_device
print(f"CUDA devices running: {device_args}")
device = torch.device(f"cuda:{device_args[0]}" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model, device_ids=device_args)
model.to(device)

# Load the SWAG dataset
train_dataset, test_dataset = get_swag_datasets(tokenizer)

# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(train_dataset)))
else:
    try:
        with open(
                os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']
    except IOError:
        with open(os.path.join(args.input_dir, args.sorting_file),
                  'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']

    train_indx = np.array(ordered_indx)
    
# Reassign train data and labels
train_dataset = train_dataset.select(train_indx)

# Get the dataloaders
train_dataloader = dataset_to_dataloader(train_dataset, args.batch_size, shuffle=True)
test_dataloader = dataset_to_dataloader(test_dataset, args.batch_size, shuffle=False)

# Setup optimizer, scheduler
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dataloader))
criterion = nn.CrossEntropyLoss(reduction='none')

# Initialize dictionary to save statistics
example_stats = {}

# Train model
elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, train_dataloader, optimizer, epoch, example_stats)
    test(args, model, device, test_dataloader, criterion, epoch, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    if args.get_stats:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        fname = os.path.join(args.output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)