#!/bin/bash

dataset=$1
ratio=$2
model_name=$3

# Get stats
python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir "results/runs/${dataset}" \
    --input_dir none \
    --get_stats \
    --sorting_file none ;

# SCDP
python src/order_examples_scdp.py \
    --embedding tfidf \
    --output_dir results \
    --ratio_list $ratio \
    --dataset $dataset \
    --key gm \
    --class_balanced 0 ;

python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "scdp_${ratio}.pkl" ;

# Random
python src/order_examples.py \
    --dataset $dataset \
    --input_dir results/runs/$dataset \
    --output_dir results \
    --random \
    --ratio_list $ratio \
    --class_balanced 1 \
    --input_fname_args seed_42__epochs_3__lr_5e-05__batch_size_32__sorting_file_none__stats_dict.pkl ;

python src/order_examples.py \
    --dataset $dataset \
    --input_dir results/runs/$dataset \
    --output_dir results \
    --ratio_list 0.3 0.5 0.7 0.9 \
    --input_fname_args seed_42__epochs_3__lr_5e-05__batch_size_32__sorting_file_none__stats_dict.pkl \
    --class_balanced 0 ;

python src/order_examples.py \
    --dataset $dataset \
    --input_dir results/runs/$dataset \
    --output_dir results \
    --ratio_list 0.3 0.5 0.7 0.9 \
    --input_fname_args seed_42__epochs_3__lr_5e-05__batch_size_32__sorting_file_none__stats_dict.pkl \
    --class_balanced 1 ;

python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "random_${ratio}.pkl" ;

# Baselines
python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "hard_monotonic_el2n_${ratio}_balance_1.pkl" ;

python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "hard_monotonic_aum_${ratio}_balance_1.pkl" ;

python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "hard_monotonic_forgetting_${ratio}_balance_1.pkl" ;

python run_glue.py \
    --dataset_name $dataset \
    --model $model_name \
    --cuda_device 0 \
    --seed 42 \
    --epochs 3 \
    --lr 5e-05 \
    --batch_size 32 \
    --output_dir results/runs/$dataset \
    --input_dir results/importance_sort/$dataset \
    --sorting_file "stratified_aum_${ratio}_balance_0.pkl"