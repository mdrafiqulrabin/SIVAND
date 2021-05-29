#!/usr/bin/env bash

model="/scratch/rabin/models/code2vec/main/java-large/saved_model_iter3"

g_root_path="/scratch/rabin/deployment/root-simplify/sm-code2vec"
cd ${g_root_path}

rm -rf data/sm/ data/tmp/ dd_data/
mkdir -p data/sm/ data/tmp/ dd_data/

echo "" > data/sm/sm.test.c2v
echo "" > data/tmp/sm_test.java

python3 dd_code2vec.py --load ${model} --test data/sm/sm.test.c2v

rm -rf data/sm/ data/tmp/
rm -rf tmp/ __pycache__/
