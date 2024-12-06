import argparse
import numpy as np
import os
import pickle
import torch
from datasets import load_dataset
from tqdm import tqdm

from coreset import CoresetSelection


# Calculates forgetting statistics per example
#
# diag_stats: dictionary created during training containing 
#             loss, accuracy, and missclassification margin 
#             per example presentation
# npresentations: number of training epochs
#
# Returns 4 dictionaries with statistics per example
#
def compute_forgetting_statistics(diag_stats, get_stats_rate):

    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}
    el2n = {}
    aum = {}

    if get_stats_rate == 1.0:
        npresentations = len(diag_stats[0][0])
    else:
        original_npresentations = len(diag_stats[0][0])
        npresentations = int(original_npresentations * get_stats_rate)
        for k, v in tqdm(diag_stats.items()):
            diag_stats_new = [[], [], [], []]
            if not isinstance(k, str):
                for i in range(len(diag_stats[k])):
                    for j in range(npresentations):
                        diag_stats_new[i].append(diag_stats[k][i][int(j / get_stats_rate)])
                diag_stats[k] = diag_stats_new

    for example_id, example_stats in diag_stats.items():

        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):

            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation[example_id] = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

            # Find EL2N score for example
            el2n[example_id] = example_stats[3]

            # Find AUM score for example
            aum[example_id] = margins_per_presentation[example_id]

    return presentations_needed_to_learn, unlearned_per_presentation, aum, first_learned, el2n


# Sorts examples by number of forgetting counts during training, in ascending order
# If an example was never learned, it is assigned the maximum number of forgetting counts
# If multiple training runs used, sort examples by the sum of their forgetting counts over all runs
#
# unlearned_per_presentation_all: list of dictionaries, one per training run
# first_learned_all: list of dictionaries, one per training run
# npresentations: number of training epochs
#
# Returns 2 numpy arrays containing the sorted example ids and corresponding forgetting counts
#

def get_all_stats(
    unlearned_per_presentation_all, 
    first_learned_all, 
    aum_all,
    el2n_average_all,
):
    
    # Initialize lists
    example_original_order = []
    example_stats = []
    el2n_average = []
    aum_average = []
    npresentations = len(unlearned_per_presentation_all[0][0])

    for example_id in unlearned_per_presentation_all[0].keys():
        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):

            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

        el2n_average.append(np.mean(el2n_average_all[example_id]))
        aum_average.append(np.mean(aum_all[example_id]))

    print('Number of unforgettable examples: {}'.format(len(np.where(np.array(example_stats) == 0)[0])))
    return np.array(example_original_order), np.array(example_stats), np.array(aum_average), np.array(el2n_average)



# Checks whether a given file name matches a list of specified arguments
#
# fname: string containing file name
# args_list: list of strings containing argument names and values, i.e. [arg1, val1, arg2, val2,..]
#
# Returns 1 if filename matches the filter specified by the argument list, 0 otherwise
#
def check_filename(fname, args):
    if fname == args:
        return 1
    else:
        return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('--input_dir', type=str, required=True)

    parser.add_argument(
        '--input_fname_args',
        help='arguments and argument values to select input filenames, i.e. arg1 val1 arg2 val2'
    )

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--dataset', type=str, required=True)
    
    parser.add_argument(
        '--get_stats_rate',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--ratio_list',
        nargs='+',
        type=float,
    )

    parser.add_argument(
        '--class_balanced',
        type=int,
    )

    parser.add_argument(
        '--random',
        action='store_true',
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))
        
    if not os.path.exists(os.path.join(args.output_dir, 'importance_sort', args.dataset)):
        os.makedirs(os.path.join(args.output_dir, 'importance_sort', args.dataset))

    # Random sampling
    if args.random:
        if args.dataset in ['cola', 'sst2', 'mrpc', 'rte', 'qnli']:
            dataset = load_dataset("nyu-mll/glue", args.dataset)
        elif args.dataset == 'swag':
            dataset = load_dataset("swag", "regular")
        else:
            dataset = load_dataset(args.dataset)

        total_num = len(dataset['train'])

        print('Random sampling')
        for ratio in args.ratio_list:
            coreset_num = int(ratio * total_num)
            coreset_index = np.random.choice(total_num, coreset_num)
            score_index = ''
            with open(
                    os.path.join(args.output_dir, 'importance_sort', args.dataset, f"random_{ratio}" + '.pkl'),
                    'wb') as fout:
                pickle.dump({
                    'indices': coreset_index,
                    'metric_value': score_index
                }, fout)
    else:
        # Initialize lists to collect forgetting stastics per example across multiple training runs
        unlearned_per_presentation_all, first_learned_all = [], []

        for d, _, fs in os.walk(args.input_dir):
            for f in fs:
                # Find the files that match input_fname_args and compute forgetting statistics
                if f.endswith('stats_dict.pkl') and check_filename(
                        f, args.input_fname_args):
                    print('including file: ' + f)

                    # Load the dictionary compiled during training run
                    with open(os.path.join(d, f), 'rb') as fin:
                        loaded = pickle.load(fin)

                    # Compute the forgetting statistics per example for training run
                    _, unlearned_per_presentation, aum_all, first_learned, el2n_average_all = compute_forgetting_statistics(loaded, args.get_stats_rate)

                    unlearned_per_presentation_all.append(unlearned_per_presentation)
                    first_learned_all.append(first_learned)

        if len(unlearned_per_presentation_all) == 0:
            print('No input files found in {} that match {}'.format(
                args.input_dir, args.input_fname_args))
        else:
            examples, forgetting, aum, el2n = get_all_stats(
                unlearned_per_presentation_all, first_learned_all, aum_all, el2n_average_all)
            
            # Get labels of dataset
            if args.dataset in ['cola', 'sst2', 'mrpc', 'rte', 'qnli']:
                dataset = load_dataset("nyu-mll/glue", args.dataset)
            elif args.dataset == 'swag':
                dataset = load_dataset("swag", "regular")
            else:
                dataset = load_dataset(args.dataset)

            if args.dataset in ['cola', 'swag', 'sst2', 'mrpc', 'rte', 'qnli']:
                train_dataset_targets = np.array(dataset["train"]['label'])
                targets = train_dataset_targets[examples]

                data_score = {
                    'examples': torch.tensor(examples),
                    'forgetting': torch.tensor(forgetting),
                    'el2n': torch.tensor(el2n),
                    'aum': torch.tensor(aum),
                    'targets': torch.tensor(targets)
                }

                total_num = data_score['targets'].shape[0]
            else:
                data_score = {
                    'examples': torch.tensor(examples),
                    'forgetting': torch.tensor(forgetting),
                    'el2n': torch.tensor(el2n),
                    'aum': torch.tensor(aum)
                }
                train_dataset = dataset['train']
                total_num = len(train_dataset)

            # Monotonic sampling
            for prune_difficulty in ['easy', 'hard']:
                for key in ['forgetting', 'el2n', 'aum']:
                    for ratio in args.ratio_list:
                        if prune_difficulty == 'easy':
                            if key == 'forgetting' or key == 'el2n':
                                descending = False
                            elif key == 'aum':
                                descending = True
                        elif prune_difficulty == 'hard':
                            if key == 'forgetting' or key == 'el2n':
                                descending = True
                            elif key == 'aum':
                                descending = False

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
                                os.path.join(args.output_dir, 'importance_sort', args.dataset, f"{prune_difficulty}_monotonic_{key}_{ratio}_balance_{args.class_balanced}" + '.pkl'),
                                'wb') as fout:
                            pickle.dump({
                                'indices': coreset_index,
                                'metric_value': score_index
                            }, fout)

            # Stratified sampling
            for ratio in args.ratio_list:
                for key in ['forgetting', 'el2n', 'aum']:
                    mis_num = 0 # Zero for now
            
                    data_score, score_index = CoresetSelection.mislabel_mask(
                        data_score,
                        mis_key='aum',
                        mis_num=mis_num,
                        mis_descending=False,
                        coreset_key=key)
                    
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
                            os.path.join(args.output_dir, 'importance_sort', args.dataset, f"stratified_{key}_{ratio}_balance_{args.class_balanced}" + '.pkl'),
                            'wb') as fout:
                        pickle.dump({
                            'indices': coreset_index,
                            'metric_value': score_index
                        }, fout)