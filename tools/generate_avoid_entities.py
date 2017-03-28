#!/usr/bin/env

import sys

import os

""" Generate a list of entities that are not presented in train.txt
"""
entities = set()
with open(os.path.join(sys.argv[1], 'entities.txt'), 'r', encoding='utf8') as f:
    for c, line in enumerate(f):
        entities.add(line.strip())
        if c % 1000 == 0:
            print("processed %d entities" % c, end='\r')

with open(os.path.join(sys.argv[1], 'train.txt'), 'r', encoding='utf8') as f:
    for c, line in enumerate(f):
        head, tail, rel = line.strip().split('\t')

        entities.discard(head)
        entities.discard(tail)

        if c % 1000 == 0:
            print("processed % lines" % c, end='\r')

print("%d entities are not seen during training." % len(entities))

with open(os.path.join(sys.argv[1], 'avoid_entities.txt'), 'w', encoding='utf8') as f:
    for ent in entities:
        f.write(ent + "\n")
