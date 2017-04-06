#!/usr/bin/env python3

import sys

import os

""" Generate
    evaluation open and closed target files

    RUN THIS AFTER AVOID_ENTITIES.TXT IS GENERATED
"""

target_tail_dict = dict()  # head relation pair and idx in target_tails
target_head_dict = dict()

open_entities = set()
with open(os.path.join(sys.argv[1], 'avoid_entities.txt'), 'r', encoding='utf8') as f:
    for line in f:
        open_entities.add(line.strip())

for input_file in ['train.txt', 'valid.txt', 'test.txt']:
    p = os.path.join(sys.argv[1], input_file)
    if not os.path.exists(p):
        print("skip %s" % p)
        continue
    with open(p, 'r', encoding='utf8') as f:
        for c, line in enumerate(f):
            head, tail, rel = line.strip().split('\t')
            hr_key = '\t'.join([head, rel])
            tr_key = '\t'.join([tail, rel])

            if hr_key not in target_tail_dict:
                target_tail_dict[hr_key] = {'open': set(), 'closed': set()}
            if tail in open_entities:
                target_tail_dict[hr_key]['open'].add(tail)
            else:
                target_tail_dict[hr_key]['closed'].add(tail)

            if tr_key not in target_head_dict:
                target_head_dict[tr_key] = {'open': set(), 'closed': set()}
            if head in open_entities:
                target_head_dict[tr_key]['open'].add(head)
            else:
                target_head_dict[tr_key]['closed'].add(head)

            if c % 5000 == 0:
                print("processed %d lines" % c, end='\r')

print("")

for value_file, idx_file, d in zip(['eval.tails.values', 'eval.heads.values'],
                                   ['eval.tails.idx', 'eval.heads.idx'],
                                   [target_tail_dict, target_head_dict]):
    with open(os.path.join(sys.argv[1], value_file+'.open'), 'w', encoding='utf8') as f_open_targets:
        with open(os.path.join(sys.argv[1], value_file+".closed"), 'w', encoding='utf8') as f_closed_targets:
            with open(os.path.join(sys.argv[1], idx_file), 'w', encoding='utf8') as f_idx:
                for k, v in d.items():
                    f_idx.write(k + "\n")
                    f_open_targets.write(" ".join(v['open']) + "\n")
                    f_closed_targets.write(" ".join(v['closed']) + "\n")
