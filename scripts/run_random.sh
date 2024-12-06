#!/bin/bash

dataset=$1
ratio=$2

if [ $dataset == "swag" ]; then
    python src/order_examples.py \
        --dataset $dataset \
        --input_dir results/runs/$dataset \
        --output_dir results \
        --random \
        --ratio_list $ratio \
        --input_fname_args seed_42__epochs_3__lr_5e-05__batch_size_32__sorting_file_none__stats_dict.pkl ;

    python run_swag.py \
        --cuda_device 0 \
        --seed 42 \
        --epochs 3 \
        --lr 5e-05 \
        --batch_size 32 \
        --output_dir results/runs/$dataset \
        --input_dir results/importance_sort/$dataset \
        --sorting_file "random_${ratio}.pkl" ;
else
    python src/order_examples.py \
        --dataset $dataset \
        --input_dir results/runs/$dataset \
        --output_dir results \
        --random \
        --ratio_list $ratio \
        --class_balanced 1 \
        --input_fname_args seed_42__epochs_3__lr_5e-05__batch_size_32__sorting_file_none__stats_dict.pkl ;

    python run_glue.py \
        --dataset_name $dataset \
        --cuda_device 0 \
        --seed 42 \
        --epochs 3 \
        --lr 5e-05 \
        --batch_size 32 \
        --output_dir results/runs/$dataset \
        --input_dir results/importance_sort/$dataset \
        --sorting_file "random_${ratio}.pkl" ;
fi