#!/bin/bash
big=( "Dutch" "English" "French" "German" "Italian" "Russian" "Spanish" "Indonesian" "Croatian" )

small=( "nl" "en" "fr" "de" "it" "ru" "es" "id" "hr" )

# big=( "Dutch" )
# small=( "nl" )

for idx in ${!big[*]}; do
    python runexp_EM.py --learning_type sup --labeled_set ../UD_${big[$idx]}/ud_${small[$idx]}_pos_char_w2v.pkl.gz --percent_labeled 20
done


for i in $(seq 25 5 80)
do
    echo "20 labeled data and $i un-labeled data"
    python runexp_EM.py --learning_type semi --labeled_set ../UD_English/ud_en_pos_char_w2v.pkl.gz --percent_labeled 20 --percent_unlabeled $i
done
