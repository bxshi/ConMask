import tensorflow as tf


def normalized_lookup(params, ids, name=None):
    with tf.name_scope(name, 'normalized_lookup'):
        _embedding = tf.nn.embedding_lookup(params, ids)
        normalized_embedding = tf.truediv(_embedding, tf.cast(tf.shape(params)[1], tf.float32))
        return normalized_embedding
