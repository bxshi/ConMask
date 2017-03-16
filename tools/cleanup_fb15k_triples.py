#!/usr/bin/env python3
import sys


# We remove triples that does not have a valid description in the DKRL paper's entity set.

# ./cleanup_fb15k_triples ENTITIES.TXT OLD_TRIPLE_FILE NEW_TRIPLE_FILE

def load_list(file_path):
    d = dict()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            elem = line.strip().split('\t')[0]
            if elem not in d:
                d[elem] = len(d)
    print("Loaded %d elements from %s" % (len(d), file_path))
    return d


entities = load_list(sys.argv[1])

with open(sys.argv[2], 'r', encoding='utf8') as f:
    with open(sys.argv[3], 'w', encoding='utf8') as fout:
        for line in f:
            src, dst, rel = line.strip().split('\t')
            if src in entities and dst in entities:
                fout.write(line)
