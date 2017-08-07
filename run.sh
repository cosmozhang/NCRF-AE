#!/bin/bash


echo "20 labeled data and 50 un-labeled data"
python runsemi_EM.py --labeled_set ./ud_en_pos_char_w2v.pkl.gz --percent_labeled 20 --percent_unlabeled 50
