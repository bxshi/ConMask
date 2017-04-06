#!/usr/bin/env python3

import sys

import os

""" Generate
    training_target_tail_file, training_target_tail_key_file
    training_target_head_file, training_target_head_key_file
"""

train_path = os.path.join(sys.argv[1], 'train.txt')

target_tail_dict = dict()  # head relation pair and idx in target_tails
target_head_dict = dict()

with open(train_path, 'r', encoding='utf8') as f:
    for c, line in enumerate(f):
        head, tail, rel = line.strip().split('\t')
        hr_key = '\t'.join([head, rel])
        tr_key = '\t'.join([tail, rel])

        if hr_key not in target_tail_dict:
            target_tail_dict[hr_key] = set()
        target_tail_dict[hr_key].add(tail)

        if tr_key not in target_head_dict:
            target_head_dict[tr_key] = set()
        target_head_dict[tr_key].add(head)

        if c % 5000 == 0:
            print("processed %d lines" % c, end='\r')

print("")

for value_file, idx_file, d in zip(['train.tails.values', 'train.heads.values'],
                                   ['train.tails.idx', 'train.heads.idx'],
                                   [target_tail_dict, target_head_dict]):
    with open(os.path.join(sys.argv[1], value_file), 'w', encoding='utf8') as f_tail:
        with open(os.path.join(sys.argv[1], idx_file), 'w', encoding='utf8') as f_idx:
            for k, v in d.items():
                f_idx.write(k + "\n")
                f_tail.write(" ".join(v) + "\n")