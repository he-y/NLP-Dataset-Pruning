#!/bin/bash

dataset=$1
ratio=$2
embedding=$3

if [ $dataset == "swag" ]; then
    python src/order_examples_scdp.py \
        --embedding $embedding \
        --output_dir results \
        --ratio_list $ratio \
        --dataset $dataset \
        --key gm \
        --class_balanced 0 ;

    python run_swag.py \
        --cuda_device 0 \
        --seed 42 \
        --epochs 3 \
        --lr 5e-05 \
        --batch_size 32 \
        --output_dir results/runs/$dataset \
        --input_dir results/importance_sort/$dataset \
        --sorting_file "scdp_${ratio}.pkl" ;
else
    python src/order_examples_scdp.py \
        --embedding $embedding \
        --output_dir results \
        --ratio_list $ratio \
        --dataset $dataset \
        --key gm \
        --class_balanced 0 ;
        
    python run_glue.py \
        --dataset_name $dataset \
        --cuda_device 0 \
        --seed 42 \
        --epochs 3 \
        --lr 5e-05 \
        --batch_size 32 \
        --output_dir results/runs/$dataset \
        --input_dir results/importance_sort/$dataset \
        --sorting_file "scdp_${ratio}.pkl" ;
fi