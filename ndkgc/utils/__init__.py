import numpy as np

import tensorflow as tf


def count_line(file_path):
    counter = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for _ in f:
            counter += 1
    return counter


def valid_vocab_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        zero_index_word = f.readline().strip().lower()
        if zero_index_word != '__pad__':
            raise ValueError(
                "The first element in vocab file %s must be __PAD__, now is %s" % (file_path, zero_index_word))


def load_rev_list(file_path):
    d = dict()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            elem = line.strip().split('\t')[0]
            i = len(d)
            d[i] = elem
    tf.logging.info("Loaded %d elements from %s" % (len(d), file_path))
    return d


def load_list(file_path):
    d = dict()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            elem = line.strip().split('\t')[0]
            if elem not in d:
                d[elem] = len(d)
    tf.logging.info("Loaded %d elements from %s" % (len(d), file_path))
    return d


def load_train_entities(entity_file, avoid_entity_file):
    entities = set()
    avoid_entities = set()
    with open(entity_file, 'r', encoding='utf8') as f:
        for line in f:
            entities.add(line.strip())
    with open(avoid_entity_file, 'r', encoding='utf8') as f:
        for line in f:
            avoid_entities.add(line.strip())
    return set.difference(entities, avoid_entities)


def load_target_file(file_path):
    target_list = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            targets = line.strip()
            target_list.append(targets)

    return target_list


def load_triples(file_path, entities=None, relations=None):
    triples = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            src, dst, rel = line.strip().split('\t')
            if entities is None and relations is None:
                triples.append([src, rel, dst])
            else:
                triples.append([entities[src], relations[rel], entities[dst]])
    tf.logging.info("Loaded %d triples from %s" % (len(triples), file_path))

    return triples


def load_pretrained_embedding(pretrained_file_path, vocab, word_embedding_size, oov):
    current_embedding = np.random.uniform(-1, 1, [len(vocab) + oov, word_embedding_size])
    current_embedding[0, :] = 0.
    with open(pretrained_file_path, 'r', encoding='utf8') as f:
        for line in f:
            elems = line.strip().split('\t')
            word = elems[0]
            if word in vocab:
                embed = elems[1:]
                current_embedding[vocab[word]] = embed
    return current_embedding


def load_content(content_file_path, entities, max_content_len=256):
    # TODO: Correctness: the max_content_len is used for limiting the memory consumption, and therefore
    # in this case we may have words that are in the vocab but we never trained on them.
    # TODO: Performance: Move the reindexing to pre-processing due to the high cost
    # for example we could use a separate C++ program to do the validation
    d = dict()
    l = dict()
    with open(content_file_path, 'r', encoding='utf8') as f:
        for line in f:
            ent, desc_len, desc = line.strip().split('\t')
            desc_len = int(desc_len)
            if ent in entities:
                if desc_len > max_content_len:
                    desc = " ".join(desc.split()[:max_content_len])
                d[entities[ent]] = desc
                l[entities[ent]] = min(max_content_len, desc_len)
    content = list()
    content_len = list()
    for i in range(len(d)):
        content.append(d[i])
        content_len.append(l[i])
    tf.logging.info("Load %d content data from %s" % (len(d), content_file_path))
    return content, content_len


def load_vocab_file(vocab_file_path):
    vocab = dict()
    with open(vocab_file_path, 'r', encoding='utf8') as f:
        for line in f:
            vocab[line.strip()] = len(vocab) - 1
    tf.logging.info("Load %d vocabs" % len(vocab))
    return vocab


def load_vocab_embedding(embedding_path, vocab_dict, oov):
    word_embedding = np.random.uniform(-np.sqrt(6) / 200, np.sqrt(6) / 200, size=[len(vocab_dict) + oov, 200])
    with open(embedding_path, 'r', encoding='utf8') as f:
        for line in f:
            elems = line.strip().split()
            ent_name = elems[0]
            vals = [float(x) for x in elems[1:]]
            if ent_name in vocab_dict:
                word_embedding[vocab_dict[ent_name]] = vals

    return word_embedding


def load_closed_manual_evaluation_file_by_rel(file_path, avoid_file_path, eval_tail=True):
    """
    Load normal head, tail, rel files, divide them into dicts
    {
        relation : {
            head : [tails]
        }
    }
    We also skip all tail entities that does not exist
    in the training file using the avoid file.

    The head are open entities only which are the entities
    that are not in the KG during training

    :param file_path:
    :param avoid_file_path:
    :return:
    """

    avoid_entities = set()
    with open(avoid_file_path, 'r', encoding='utf8') as f:
        for line in f:
            avoid_entities.add(line.strip())
    tf.logging.info("avoid entities %d" % len(avoid_entities))
    eval_triple_dict = dict()  # rel : {head : [tail]}
    skip = 0
    loaded = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            head, tail, rel = line.strip().split('\t')
            # Only evaluate unseen entity -> seen entity
            if not eval_tail:
                tmp = head
                head = tail
                tail = tmp
            if tail in avoid_entities or head in avoid_entities:
                skip += 1
                continue
            loaded += 1
            if rel not in eval_triple_dict:
                eval_triple_dict[rel] = dict()
            if head not in eval_triple_dict[rel]:
                eval_triple_dict[rel][head] = {tail}
            else:
                eval_triple_dict[rel][head].add(tail)
    tf.logging.info("SKipped %d/%d testing cases" % (skip, skip + loaded))
    return eval_triple_dict


def load_manual_evaluation_file_by_rel(file_path, avoid_file_path, eval_tail=True):
    """
    Load normal head, tail, rel files, divide them into dicts
    {
        relation : {
            head : [tails]
        }
    }
    We also skip all tail entities that does not exist
    in the training file using the avoid file.

    The head are open entities only which are the entities
    that are not in the KG during training

    :param file_path:
    :param avoid_file_path:
    :return:
    """

    avoid_entities = set()
    with open(avoid_file_path, 'r', encoding='utf8') as f:
        for line in f:
            avoid_entities.add(line.strip())
    tf.logging.info("avoid entities %d" % len(avoid_entities))
    eval_triple_dict = dict()  # rel : {head : [tail]}
    skip = 0
    loaded = 0
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            head, tail, rel = line.strip().split('\t')
            # Only evaluate unseen entity -> seen entity
            if not eval_tail:
                tmp = head
                head = tail
                tail = tmp
            if tail in avoid_entities or head not in avoid_entities:
                skip += 1
                continue
            loaded += 1
            if rel not in eval_triple_dict:
                eval_triple_dict[rel] = dict()
            if head not in eval_triple_dict[rel]:
                eval_triple_dict[rel][head] = {tail}
            else:
                eval_triple_dict[rel][head].add(tail)
    tf.logging.info("SKipped %d/%d testing cases" % (skip, skip + loaded))
    return eval_triple_dict


def load_manual_evaluation_file(file_path, avoid_file_path):
    """
    Load normal head, tail, rel files, divide them into
        dicts where the key is unique head rel pair and
        the values are the targets in the evaluation set.

    We also skip all tail entities that does not exist in
    the training file.

    The head are open entities only

    :param file_path: test.txt file
    :return:
    """
    avoid_entities = set()
    with open(avoid_file_path, 'r', encoding='utf8') as f:
        for line in f:
            avoid_entities.add(line.strip())

    eval_triple_dict = dict()  # 'head \t rel' : [tail]
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            head, tail, rel = line.strip().split('\t')
            if tail in avoid_entities or head not in avoid_entities:
                continue
            k = "\t".join([head, rel])
            if k not in eval_triple_dict:
                eval_triple_dict[k] = {tail}
            else:
                eval_triple_dict[k].add(tail)
    return eval_triple_dict


def load_relation_specific_targets(tail_rel_path, rel_path):
    """
    Based on training file, generate a subset of targets for each
        relationship. The assumption is if entity has relationship R
        before then it might be a valid target.
    :param tail_rel_path:
    :param rel_path:
    :return:
    """
    rels = dict()
    with open(rel_path, 'r', encoding='utf8') as f:
        for line in f:
            rels[line.strip()] = set()
    with open(tail_rel_path, 'r', encoding='utf8') as f:
        for line in f:
            tail, rel = line.strip().split('\t')
            rels[rel].add(tail)
    return rels


def load_filtered_targets(idx_file, value_file):
    filtered_targets = dict()  # ent\t rel -> list of targets
    with open(idx_file, 'r', encoding='utf8') as f_idx:
        with open(value_file, 'r', encoding='utf8') as f_val:
            for idx, val in zip(f_idx, f_val):
                filtered_targets[idx.strip()] = val.strip().split()

    return filtered_targets
