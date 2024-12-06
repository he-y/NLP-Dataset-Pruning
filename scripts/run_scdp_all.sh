#!/bin/bash

bash scripts/run_scdp.sh rte 0.3 tfidf
bash scripts/run_scdp.sh rte 0.5 tfidf
bash scripts/run_scdp.sh rte 0.7 tfidf
bash scripts/run_scdp.sh rte 0.9 tfidf

bash scripts/run_scdp.sh mrpc 0.3 tfidf
bash scripts/run_scdp.sh mrpc 0.5 tfidf
bash scripts/run_scdp.sh mrpc 0.7 tfidf
bash scripts/run_scdp.sh mrpc 0.9 tfidf

bash scripts/run_scdp.sh cola 0.3 tfidf
bash scripts/run_scdp.sh cola 0.5 tfidf
bash scripts/run_scdp.sh cola 0.7 tfidf
bash scripts/run_scdp.sh cola 0.9 tfidf

bash scripts/run_scdp.sh sst2 0.3 tfidf
bash scripts/run_scdp.sh sst2 0.5 tfidf
bash scripts/run_scdp.sh sst2 0.7 tfidf
bash scripts/run_scdp.sh sst2 0.9 tfidf

bash scripts/run_scdp.sh swag 0.3 tfidf
bash scripts/run_scdp.sh swag 0.5 tfidf
bash scripts/run_scdp.sh swag 0.7 tfidf
bash scripts/run_scdp.sh swag 0.9 tfidf

bash scripts/run_scdp.sh qnli 0.3 tfidf
bash scripts/run_scdp.sh qnli 0.5 tfidf
bash scripts/run_scdp.sh qnli 0.7 tfidf
bash scripts/run_scdp.sh qnli 0.9 tfidf