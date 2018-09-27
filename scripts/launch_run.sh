#!/bin/bash -e

cd ../
source activate cmput659
python run.py --idx $1 --config-file $2
