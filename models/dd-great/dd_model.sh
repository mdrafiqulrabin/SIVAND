#!/usr/bin/env bash

g_root_path="/scratch/rabin/deployment/root-simplify/sm-great"
cd ${g_root_path}

rm -rf sm_data/dd_data/
mkdir -p sm_data/dd_data/ sm_data/tmp/great/eval/
echo "" > sm_data/tmp/great/eval/tmp.txt

cd "running/"
python3 dd_model.py

cd ${g_root_path}
rm -rf sm_data/tmp/
