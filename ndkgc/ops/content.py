import tensorflow as tf
import tensorflow.contrib.lookup as lookup


def get_content_matrix(variable_scope, size, reuse=True, device='/cpu:0'):
    """ Return content matrix

    :param variable_scope:
    :param size:
    :param reuse:
    :param device:
    :return:
    """
    with tf.device(device):
        with tf.variable_scope(variable_scope, reuse=reuse):
            return tf.get_variable('content', shape=[size],
                                   dtype=tf.string,
                                   trainable=False,
                                   collections="static_variables")


def get_lookup_table(element_file_path, oov_buckets, size=None, device='/cpu:0', name='lookup_table'):
    with tf.device(device):
        return lookup.string_to_index_table_from_file(vocabulary_file=element_file_path,
                                                      num_oov_buckets=oov_buckets,
                                                      vocab_size=size,
                                                      default_value=0,  # 0 is always the padding value
                                                      name=name)


def content_lookup(content, vocab_table, id, name=None):
    """ Lookup a single entity's content from `content`, and convert words into ids using vocab_table.

    :param content: A 1-D string matrix
    :param vocab_table: A tensorflow table
    :param id: A scalar
    :param name:
    :return: Extract ids, 1-d vector
    """
    with tf.name_scope(name, 'content_lookup', [content, vocab_table, id]):
        extracted_content = tf.string_split([content[id]], delimiter=' ')
        if isinstance(extracted_content, tf.SparseTensor):
            return vocab_table.lookup(extracted_content.values)
        else:
            return vocab_table.lookup(extracted_content)


def multiple_content_lookup(content, vocab_table, ids, name=None):
    """

    :param content:
    :param vocab_table:
    :param ids:
    :param name:
    :return: 2-D [batch_size, max_length_in_batch] content id matrix,
             1-D [batch_size] content len vector
    """
    with tf.name_scope(name, 'multiple_content_lookup', [content, vocab_table, ids]):
        content_list = tf.nn.embedding_lookup(content, ids)

        extracted_sparse_content = tf.string_split(content_list, delimiter=' ')

        sparse_content = tf.SparseTensor(indices=extracted_sparse_content.indices,
                                         values=vocab_table.lookup(extracted_sparse_content.values),
                                         dense_shape=extracted_sparse_content.dense_shape)

        extracted_content_ids = tf.sparse_tensor_to_dense(sparse_content,
                                                          default_value=0, name='dense_content')
        extracted_content_len = tf.reduce_sum(tf.cast(tf.not_equal(extracted_content_ids, 0), tf.int32), axis=-1)

        return extracted_content_ids, extracted_content_len
