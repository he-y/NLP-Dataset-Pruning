import argparse
import numpy as np
import os
import pickle
import torch
from datasets import load_dataset
from feature_extraction import tfidf_dist, sbert_dist
from utils import convert_datasets_keys
from coreset import CoresetSelection

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ratio_list', type=float, nargs='+', required=True)
    parser.add_argument('--class_balanced', type=int, required=True)
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--key', type=str)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, 'importance_sort', args.dataset)):
        os.makedirs(os.path.join(args.output_dir, 'importance_sort', args.dataset))

    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

    sentence1_key, sentence2_key = convert_datasets_keys(args.dataset)

    if args.dataset in ['cola', 'sst2', 'mrpc', 'rte', 'qnli']:
        dataset = load_dataset("nyu-mll/glue", args.dataset)
    elif args.dataset == 'swag':
        dataset = load_dataset("swag", "regular")

    if sentence2_key:
        train_dataset_documents = [f"{x[sentence1_key]} {x[sentence2_key]}" for x in dataset["train"]]
    else:
        train_dataset_documents = [f"{x[sentence1_key]}" for x in dataset["train"]]
        
    train_dataset_targets = np.array(dataset["train"]['label'])
    targets = train_dataset_targets

    if args.embedding == 'tfidf':
        dist = tfidf_dist(train_dataset_documents, dist_type=args.key)
    elif args.embedding == 'sentence_bert':
        dist = sbert_dist(train_dataset_documents, 
                            model='sentence-transformers/all-MiniLM-L12-v2', 
                            dist_type=args.key)

    data_score = {
        'examples': np.arange(len(train_dataset_documents)),
        f'{args.key}': dist.clone().detach().requires_grad_(False),
        'targets': torch.tensor(targets)
    }

    total_num = data_score['targets'].shape[0]

    # Monotonic sampling
    for prune_difficulty in ['hard']:
        key = args.key
        for ratio in args.ratio_list:
            if ratio * total_num <= 1500:
                descending = False if prune_difficulty == 'easy' else True

                coreset_index, score_index = CoresetSelection.score_monotonic_selection(
                    data_score=data_score,
                    key=key,
                    ratio=ratio,
                    descending=descending,
                    class_balanced=bool(args.class_balanced),
                    total_num=total_num
                )

                # Coreset index is the index in data_score, just need to get original index in dataset in examples
                
                coreset_index_fnl = []
                for i in coreset_index:
                    idx = data_score['examples'][i]
                    coreset_index_fnl.append(idx)
                
                coreset_index = np.array(coreset_index_fnl)

                # Save outputs
                with open(
                        os.path.join(args.output_dir, 'importance_sort', args.dataset, f"scdp_{ratio}" + '.pkl'),
                        'wb') as fout:
                    pickle.dump({
                        'indices': coreset_index,
                        'metric_value': score_index
                    }, fout)

    # Stratified sampling

    for ratio in args.ratio_list:
        if ratio * total_num > 1500:
            key = args.key
            
            coreset_num = int(ratio * total_num)

            if bool(args.class_balanced):
                coreset_index, score_index = CoresetSelection.class_balanced_stratified_sampling(
                    data_score=data_score,
                    coreset_key=key,
                    coreset_num=coreset_num)
            else:
                coreset_index, score_index = CoresetSelection.stratified_sampling(
                    data_score=data_score, 
                    coreset_key=key,
                    coreset_num=coreset_num)
                
            coreset_index_fnl = []
            for i in coreset_index:
                idx = data_score['examples'][i]
                coreset_index_fnl.append(idx)
            
            coreset_index = np.array(coreset_index_fnl)

            # Save outputs
            with open(
                    os.path.join(args.output_dir, 'importance_sort', args.dataset, f"scdp_{ratio}" + '.pkl'),
                    'wb') as fout:
                pickle.dump({
                    'indices': coreset_index,
                    'metric_value': score_index
                }, fout)