def convert_datasets_keys(dataset_name):
    if dataset_name == 'qqp':
        sentence1_key = 'question1'
        sentence2_key = 'question2'
    elif dataset_name == 'mnli':
        sentence1_key = 'premise'
        sentence2_key = 'hypothesis'
    elif dataset_name == 'sst2':
        sentence1_key = 'sentence'
        sentence2_key = None
    elif dataset_name == 'cola':
        sentence1_key = 'sentence'
        sentence2_key = None
    elif dataset_name == 'mrpc':
        sentence1_key = 'sentence1'
        sentence2_key = 'sentence2'
    elif dataset_name == 'qnli':
        sentence1_key = 'question'
        sentence2_key = 'sentence'
    elif dataset_name == 'rte':
        sentence1_key = 'sentence1'
        sentence2_key = 'sentence2'
    elif dataset_name == 'wnli':
        sentence1_key = 'sentence1'
        sentence2_key = 'sentence2'
    elif dataset_name == 'stsb':
        sentence1_key = 'sentence1'
        sentence2_key = 'sentence2'
    elif dataset_name == 'imdb':
        sentence1_key = 'text'
        sentence2_key = None
    elif dataset_name == 'swag':
        sentence1_key = 'sent1'
        sentence2_key = 'sent2'

    return sentence1_key, sentence2_key