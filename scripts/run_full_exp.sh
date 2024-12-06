#!/bin/bash

dataset=$1

bash scripts/run_random.sh $dataset 0.3
bash scripts/run_random.sh $dataset 0.5
bash scripts/run_random.sh $dataset 0.7
bash scripts/run_random.sh $dataset 0.9

bash scripts/run_el2n.sh $dataset 0.3
bash scripts/run_el2n.sh $dataset 0.5
bash scripts/run_el2n.sh $dataset 0.7
bash scripts/run_el2n.sh $dataset 0.9

bash scripts/run_aum.sh $dataset 0.3
bash scripts/run_aum.sh $dataset 0.5
bash scripts/run_aum.sh $dataset 0.7
bash scripts/run_aum.sh $dataset 0.9

bash scripts/run_forgetting.sh $dataset 0.3
bash scripts/run_forgetting.sh $dataset 0.5
bash scripts/run_forgetting.sh $dataset 0.7
bash scripts/run_forgetting.sh $dataset 0.9

bash scripts/run_ccs.sh $dataset 0.3
bash scripts/run_ccs.sh $dataset 0.5
bash scripts/run_ccs.sh $dataset 0.7
bash scripts/run_ccs.sh $dataset 0.9