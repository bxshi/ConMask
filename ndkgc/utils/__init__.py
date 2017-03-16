import numpy as np


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


def load_list(file_path):
    d = dict()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            elem = line.strip().split('\t')[0]
            if elem not in d:
                d[elem] = len(d)
    print("Loaded %d elements from %s" % (len(d), file_path))
    return d


def load_triples(file_path, entities, relations):
    triples = list()
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            src, dst, rel = line.strip().split('\t')
            triples.append([entities[src], relations[rel], entities[dst]])
    print("Loaded %d triples from %s" % (len(triples), file_path))

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


def load_content(content_file_path, entities):
    d = dict()
    with open(content_file_path, 'r', encoding='utf8') as f:
        for line in f:
            ent, desc = line.strip().split('\t')
            if ent in entities:
                d[entities[ent]] = desc
    content = list()
    for i in range(len(d)):
        content.append(d[i])
    print("Load %d content data from %s" % (len(d), content_file_path))
    return np.asarray(content)
