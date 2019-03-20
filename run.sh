#!/bin/bash

mkdir -p data
mkdir -p report
mkdir -p report/one_model

python cross_validate.py bHLH
python cross_validate.py ETS
python cross_validate.py E2F
python cross_validate.py RUNX

python cross_validate_one_model.py

