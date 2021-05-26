#!/usr/bin/env bash

model="/scratch/rabin/models/code2seq/main/java-large/saved_model_iter52"

g_root_path="/scratch/rabin/deployment/root-simplify/sm-code2seq"
cd ${g_root_path}

rm -rf data/sm/ data/tmp/ attn_data/
mkdir -p data/sm/ data/tmp/ attn_data/

echo "" > data/sm/sm.test.c2s
echo "" > data/tmp/sm_test.java

python3 attn_code2seq.py --load ${model} --test data/sm/sm.test.c2s

rm -rf data/sm/ data/tmp/
rm -rf tmp/ __pycache__/
