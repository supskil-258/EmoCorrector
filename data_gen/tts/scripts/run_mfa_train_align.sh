#!/bin/bash

BASE_DIR=processed/ECD-TSE
MODEL_MFA=checkpoints

mkdir -p $BASE_DIR/mfa_outputs_tmp/clean

mfa align $BASE_DIR/mfa_inputs/clean/ $BASE_DIR/clean_mfa_dict.txt $MODEL_MFA/mfa_model.zip $BASE_DIR/mfa_outputs_tmp/clean -t $BASE_DIR/mfa_tmp/clean   --clean -j 128

mkdir -p $BASE_DIR/mfa_outputs/clean

find $BASE_DIR/mfa_outputs_tmp/clean -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/clean
