import os
import sys
import argparse
import time
import pickle
import logging
import pprint
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as npr
import evaluate

from src.model import get_distilbert_model, get_bert_model, BertClass
from src.preprocess_glue import *
from src.utils import convert_datasets_keys

parser = argparse.ArgumentParser(description='Training GLUE')

parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    metavar='N',
)

parser.add_argument(
    '--cuda_device',
    type=int,
    nargs='+',
    default=[0],
)

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S'
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
    '--input_dir',
)

parser.add_argument(
    '--get_stats',
    action='store_true',
)

parser.add_argument(
    '--output_dir', 
    required=True,
)

parser.add_argument(
    '--dataset_name', 
    required=True,
)

parser.add_argument(
    '--model',
    required=False,
    default='distilbert',
)

# Arguments that you want to be in the filename of the saved output
ordered_args = [
    'seed', 
    'epochs', 
    'lr', 
    'batch_size', 
    'sorting_file',
]

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()

if not os.path.exists(f'logs/{args.dataset_name}'):
    os.makedirs(f'logs/{args.dataset_name}')

args_dict = vars(args)
save_fname = '__'.join('{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set logging configuration
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'logs/{args.dataset_name}/{save_fname}.log', "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create a FileHandler that opens the file in 'w' mode (write, truncate existing)
file_handler = logging.FileHandler(f'logs/{args.dataset_name}/{save_fname}.log', mode='w')
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
        criterion,
        optimizer,
        epoch, 
        example_stats
    ):
    train_loss = 0
    correct = 0
    total = 0

    model.train()

    for batch_idx, (batch, batch_indices) in enumerate(train_dataloader):
        # Map to device
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['label']
        labels_n_hot = torch.zeros((labels.shape[0], 2)).scatter_(1, labels.view(-1, 1), 1).to(device, dtype=torch.float)

        # Forward propagation, compute loss, get predictions
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        outputs = nn.Softmax(dim=1)(logits)
        loss = criterion(outputs, labels_n_hot)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()

        # Update loss, backprop, and step optimizer
        loss = loss.mean()
        train_loss += loss.item()
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate stats on training examples
        if args.get_stats:
            # Gather stats after a number of steps
            split_dict = {
                'sst2': 1200,
                'mrpc': 90,
                'qnli': 2000,
                'cola': 180,
                'rte': 40,
            }
            if batch_idx % split_dict[args.dataset_name] == 0 or batch_idx == len(train_dataloader) - 1:
                example_stats = test_on_train_and_calc_score(args, model, device, train_dataloader, example_stats)
                elapsed_time = time.time() - start_time
                print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Batch [%3d] Loss: %.4f Acc: %.3f%%' %
            (epoch + 1, args.epochs, batch_idx + 1, loss.item(), 100. * correct / total)
        )
        sys.stdout.flush()

        # Add training loss to dict
        index_stats = example_stats.get('train', [[], []])
        index_stats[0].append(train_loss / (batch_idx + 1))
        example_stats['train'] = index_stats

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
            inputs = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['label']
            labels_n_hot = torch.zeros((labels.shape[0], 2)).scatter_(1, labels.view(-1, 1), 1).to(device, dtype=torch.float)

            # Forward propagation, get predictions
            logits = model(inputs, attention_mask)
            outputs = nn.Softmax(dim=1)(logits)
            loss = criterion(outputs, labels_n_hot)[:, 0]

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()

            # Update statistics and loss
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
    test_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (batch, batch_indices) in enumerate(test_dataloader):
            # Map to device
            inputs = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['label']
            labels_n_hot = torch.zeros((labels.shape[0], 2)).scatter_(1, labels.view(-1, 1), 1).to(device, dtype=torch.float)

            # Forward propagation, compute loss, get predictions
            logits = model(inputs, attention_mask)
            outputs = nn.Softmax(dim=1)(logits)
            loss = criterion(outputs, labels_n_hot)
            loss = loss.mean()
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            test_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()     

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.numpy())

            sys.stdout.write('\r')
            sys.stdout.write(
                '| Batch [%3d / %3d] '%
                (batch_idx + 1, len(test_dataloader)))
            sys.stdout.flush()
        
        # Calculate precision, recall, F1 score
        result = metric.compute(predictions=all_predictions, references=all_labels)
        index_stats = example_stats.get('test', [[], []])
        index_stats[0].append(test_loss / len(test_dataloader))
        index_stats[1].append(100. * correct / float(total))
        example_stats['test'] = index_stats
        print("\n| Validation Epoch #%d\tLoss: %.4f" %
                (epoch + 1, loss.item()))
        pprint.pprint(result)

#-----------------------------------------------------------------------------------------

if args.model == 'distilbert':
    tokenizer, model = get_distilbert_model()
else:
    tokenizer, model = get_bert_model(args.model)
model = BertClass(model=model)

device_args = args.cuda_device
print(f"CUDA devices running: {device_args}")
device = torch.device(f"cuda:{device_args[0]}" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model, device_ids=device_args)
model.to(device)

metric = evaluate.load("glue", args.dataset_name)

sentence1_key, sentence2_key = convert_datasets_keys(args.dataset_name)

# Load the glue dataset
train_dataset, test_dataset = get_glue_datasets(
    dataset_name=args.dataset_name,
    sentence1_key=sentence1_key,
    sentence2_key=sentence2_key,
    tokenizer=tokenizer,
    padding="max_length",
    truncation=True
)

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

# Setup criterion, optimizer
criterion = nn.BCELoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dataloader))

# Initialize dictionary to save statistics
example_stats = {}

# Train model
elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, train_dataloader, criterion, optimizer, epoch, example_stats)
    test(args, model, device, test_dataloader, criterion, epoch, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Save the stats dictionary
    if args.get_stats:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        fname = os.path.join(args.output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)