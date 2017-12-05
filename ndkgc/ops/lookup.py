import tensorflow as tf


def normalized_lookup(params, ids, name=None):
    with tf.name_scope(name, 'normalized_lookup'):
        _embedding = tf.nn.embedding_lookup(params, ids)
        normalized_embedding = tf.truediv(_embedding, tf.cast(tf.shape(params)[1], tf.float32))
        return normalized_embedding


def triple_id_lookup(heads, rels, tails, ent_table, rel_table, name=None):
    with tf.name_scope(name, 'triple_id_lookup', [heads, rels, tails, ent_table, rel_table]):
        h = ent_table.lookup(heads)
        t = ent_table.lookup(tails)
        r = rel_table.lookup(rels)

        return h, r, t


def normalized_embedding(embedding, name=None):
    with tf.name_scope(name, 'normalized_embedding'):
        # add a small epsilon to avoid divide-by-zero
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), -1, keep_dims=True), name='norm') + 1e-10
        # norm = tf.check_numerics(norm, 'norm')
        norm_embed = embedding / norm

        # return tf.check_numerics(norm_embed, 'normalized_embedding')
        return norm_embed
