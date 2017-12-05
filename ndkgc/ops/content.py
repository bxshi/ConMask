import tensorflow as tf
import tensorflow.contrib.lookup as lookup
from tensorflow.contrib.layers import xavier_initializer, batch_norm


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


def get_lookup_table_from_tensor(tensor, oov_buckets, device='/cpu:0', name='lookup_table'):
    with tf.device(device):
        return lookup.string_to_index_table_from_tensor(tensor,
                                                        num_oov_buckets=oov_buckets,
                                                        default_value=-1,
                                                        name=name)


def get_lookup_table(element_file_path, oov_buckets, size=None, device='/cpu:0', name='lookup_table'):
    with tf.device(device):
        return lookup.string_to_index_table_from_file(vocabulary_file=element_file_path,
                                                      num_oov_buckets=oov_buckets,
                                                      vocab_size=size,
                                                      default_value=-1,  # -1 is always the padding value
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
            return tf.cast(vocab_table.lookup(extracted_content.values), tf.int32)
        else:
            return tf.cast(vocab_table.lookup(extracted_content), tf.int32)


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


def entity_content_embedding_lookup(entities, content, content_len, vocab_table, word_embedding, str_pad, name=None):
    """ Lookup entity word embeddings given a flatten 1-D entity id list and content lookup table

    :param entities: Must be a 1-D entity vector
    :param content:
    :param content_len:
    :param vocab_table:
    :param word_embedding:
    :param str_pad:
    :param name:
    :return:
    """
    with tf.device('/cpu:0'):
        with tf.name_scope(name, 'entity_content_lookup',
                           [entities, content, content_len, vocab_table, word_embedding]):
            ent_content = tf.string_split(tf.nn.embedding_lookup(content, entities, name='ent_content'), delimiter=' ')
            content_len = tf.nn.embedding_lookup(content_len, entities, name='ent_content_len')
            ent_content_dense = tf.sparse_tensor_to_dense(ent_content,
                                                          default_value=str_pad,
                                                          name='ent_content_dense')
            ent_embedding = tf.nn.embedding_lookup(word_embedding,
                                                   vocab_table.lookup(ent_content_dense,
                                                                      name='ent_content_ids'))

            return ent_embedding, content_len


def avg_content(content_embedding, content_len, padding_embedding, name=None):
    """ Content embedding without padding embeddings

     Works with 3-D tensor [?, content_len, embedding_size]
     and 4-D tensor [?, ?, content_len, embedding_size].

    :param content_embedding: [?, content_len, embedding_size] or [?, ?, content_len, embedding_size]
    :param content_len: [?] or [?, ?]
    :param padding_embedding: [embedding_size] We subtract this from the embedding sum
    :param name:
    :return: [N, embedding_size]
    """
    with tf.name_scope(name, 'avg_content', [content_embedding, content_len, padding_embedding]):
        # current content len with padding
        padded_content_len = tf.shape(content_embedding)[-2]

        embedding_sum = tf.reduce_sum(content_embedding, axis=-2, name='content_embedding_sum')

        # Depending on the rank of content_embedding, pad to [1, embedding_size] or [1, 1, embedding_size]
        padding = tf.cond(tf.equal(tf.rank(content_embedding), 3),
                          lambda: tf.expand_dims(padding_embedding, axis=0),
                          lambda: tf.expand_dims(tf.expand_dims(padding_embedding, axis=0), axis=0))

        # expand to [?, 1] or [?,?, 1]
        content_len = tf.cast(tf.expand_dims(content_len, axis=-1), tf.float32)
        padding_len = tf.cast(padded_content_len, tf.float32) - content_len

        modified_embedding_sum = embedding_sum - padding * padding_len
        avg_content_embedding = modified_embedding_sum / content_len
        return avg_content_embedding


def description_and_title_lookup(entities, content, content_len,
                                 title, title_len, vocab_table, word_embedding, str_pad, name=None):
    """ A convenience function for looking up both content and title embeddings.

    This will preserve the input entity shape. So the output shape would be

     tf.shape(entities) + [length of content or title, word_embedding]

    :param entities:
    :param content:
    :param content_len:
    :param title:
    :param title_len:
    :param vocab_table:
    :param word_embedding:
    :param str_pad:
    :param name:
    :return: 2 tuples with rank tf.rank(entities) + 2 and tf.rank(entities).
     (entity_padded_content_word_embedding, entity_content_true_length) and
     (entity_padded_title_word_embedding, entity_titile_true_length)

     The shape of padded data is tf.shape(entities) + [length of content/title, word_embedding]
    """
    varlist = [entities, content, content_len,
               title, title_len, vocab_table, word_embedding]
    with tf.name_scope(name, 'desc_title_lookup', varlist):
        flatten_entities = tf.reshape(entities, [-1], 'flatten_entities')

        content_embedding, content_true_len = entity_content_embedding_lookup(flatten_entities,
                                                                              content,
                                                                              content_len,
                                                                              vocab_table,
                                                                              word_embedding,
                                                                              str_pad,
                                                                              name='desc')

        title_embedding, title_true_len = entity_content_embedding_lookup(flatten_entities,
                                                                          title,
                                                                          title_len,
                                                                          vocab_table,
                                                                          word_embedding,
                                                                          str_pad,
                                                                          name='title')

        tf.logging.debug("flatten content_embedding %s title_embedding %s" % (content_embedding,
                                                                              title_embedding))
        # reshape [-1, content length, word_embedding_size]
        # to [????, content length, word_embedding_size]
        content_embedding, title_embedding = [tf.reshape(x,
                                                         tf.concat([tf.shape(entities),
                                                                    tf.shape(x)[1:]], axis=0),
                                                         y) for x, y in
                                              zip([content_embedding, title_embedding],
                                                  ['content_word_embedding',
                                                   'title_word_embedding'])]

        tf.logging.debug("content_embedding %s title_embedding %s" % (content_embedding,
                                                                      title_embedding))

        content_true_len, title_true_len = [tf.reshape(x, tf.shape(entities), y) for x, y in
                                            zip([content_true_len, title_true_len],
                                                ['content_sequence_len',
                                                 'title_sequence_len'])]

        return (content_embedding, content_true_len), (title_embedding, title_true_len)


def mask_content_embedding(entity_embeddings, relation_embeddings, prev_window_size=5, name=None):
    """ Calculate the similarity

    :param entity_embeddings: [?, n_entities, content_length, word_embed_size]
    :param relation_embeddings: [batch_size, word_embed_size]
    :param prev_window_size: an integer about how many words we look back when
        calculating the local context similarity
    :param name:
    :return:
    """
    varlist = [entity_embeddings, relation_embeddings]

    if len(entity_embeddings.get_shape()) != 4:
        tf.logging.error("Entity embedding must have a rank of 4, get %s" % entity_embeddings)

    with tf.name_scope(name, 'mask_content', varlist):
        # get batch size from relationship because the entity embedding batch size
        # might be 1 if all inputs are sharing the same targets (this might happen
        # during evaluation.)
        # batch_size = tf.shape(relation_embeddings)[0]
        # n_entities = tf.shape(entity_embeddings)[1]
        # content_len = tf.shape(entity_embeddings)[-2]

        # expand from [batch_size, word_embed_size] to [batch_size, 1, 1, word_embed_size]
        relation_embeddings = tf.expand_dims(tf.expand_dims(relation_embeddings, axis=1), axis=1)

        # [batch_size, n_entities, content_len]
        word_similarity = tf.reduce_sum(entity_embeddings * relation_embeddings, axis=-1)

        # [batch_size, n_entities, content_len + 4]
        padded_word_similarity = tf.pad(word_similarity, [[0, 0], [0, 0], [prev_window_size - 1, 0]],
                                        name='padded_word_similarity')

        # the context_similarity score is based on the max similarity score of the current word and 4 previous words
        similarity_maxpool = tf.layers.max_pooling2d(tf.expand_dims(padded_word_similarity,
                                                                    -1),
                                                     pool_size=[1, prev_window_size],
                                                     strides=[1, 1],
                                                     padding='same',
                                                     name='similarity_maxpool')

        context_similarity = tf.slice(similarity_maxpool, [0, 0, 0, 0],
                                      # [batch_size, n_entities, content_len, 1]
                                      tf.concat([tf.shape(word_similarity), [1]], axis=0),
                                      'unscaled_context_similarity')

        context_similarity = tf.nn.sigmoid(context_similarity,
                                           'context_similarity')

        masked_content = entity_embeddings * context_similarity

        return masked_content


def extract_embedding_by_dkrl(content_embedding,
                              filters,
                              variable_scope, reuse=True, name=None):
    """ Extract an embedding for each instance.

    This is similar to DKRL, the first k-1 layers are using maxpooling and the last layer
    uses mean pooling

    :param content_embedding: [batch_size, n_entities, content_len, word_embedding]
    :param filters: Number of filters in each CNN layer
    :param variable_scope:
    :param reuse:
    :param name:
    :return:
    """

    if len(content_embedding.get_shape()) != 4:
        tf.logging.error("Content embedding must have a rank of 4, get %s" % (content_embedding))

    with tf.name_scope(name, 'extract_embedding_by_dkrl', [content_embedding]):
        with tf.variable_scope(variable_scope, reuse=reuse):
            # Reshape the input to [batch, length, channel] so the conv1d has defined shapes
            #  to declare new variables
            conv_output = tf.reshape(content_embedding,
                                     tf.stack([-1,  # batch size
                                               tf.shape(content_embedding)[-2],  # content length
                                               filters]))  # word_embed_size

            conv_output = tf.layers.conv1d(conv_output,
                                           filters=filters,
                                           kernel_size=2,
                                           strides=1,
                                           padding='same',
                                           activation=tf.nn.sigmoid,
                                           # change loss to sigmoid to see if we can avoid the nan (without relu the loss will not reduce at all)
                                           use_bias=True,
                                           kernel_initializer=xavier_initializer(),
                                           bias_initializer=xavier_initializer(),
                                           trainable=True,
                                           name='layer_1')
            # conv_output = tf.check_numerics(conv_output, conv_output.name)

            conv_output = tf.layers.max_pooling1d(conv_output,
                                                  pool_size=4,
                                                  strides=1,
                                                  padding='same',
                                                  name='layer_1_maxpool')

            conv_output = tf.layers.conv1d(conv_output,
                                           filters=filters,
                                           kernel_size=2,
                                           strides=1,
                                           padding='same',
                                           activation=tf.nn.sigmoid,
                                           # change loss to sigmoid to see if we can avoid the nan (without relu the loss will not reduce at all)
                                           use_bias=True,
                                           kernel_initializer=xavier_initializer(),
                                           bias_initializer=xavier_initializer(),
                                           trainable=True,
                                           name='layer_2')
            # conv_output = tf.check_numerics(conv_output, conv_output.name)

            conv_output = tf.reduce_mean(conv_output, axis=-2)

            conv_output = tf.reshape(conv_output,
                                     tf.stack([-1,
                                               tf.shape(content_embedding)[1],
                                               filters], axis=0),
                                     # tf.concat([tf.shape(content_embedding)[:2],
                                     #            [filters]], axis=0),
                                     name='fcn_output')

            tf.logging.debug("conv_output %s" % conv_output)

            return conv_output

            # # Reshape the input to [batch, length, channel] so the conv1d has defined shapes
            # #  to declare new variables
            # conv_output = tf.reshape(content_embedding,
            #                          tf.stack([-1,
            #                                    tf.shape(content_embedding)[-2],
            #                                    filters]))
            #
            # conv_output = tf.layers.conv1d(conv_output,
            #                                filters=filters,
            #                                kernel_size=2,
            #                                strides=1,
            #                                padding='same',
            #                                activation=tf.nn.sigmoid,
            #                                use_bias=True,
            #                                kernel_initializer=xavier_initializer(),
            #                                trainable=True,
            #                                name='layer_1')
            # conv_output = tf.layers.max_pooling1d(conv_output,
            #                                       pool_size=4,
            #                                       strides=1,
            #                                       name='layer_1_maxpool')
            # conv_output = tf.layers.conv1d(conv_output,
            #                                filters=filters,
            #                                kernel_size=1,
            #                                strides=1,
            #                                padding='same',
            #                                activation=tf.nn.sigmoid,
            #                                use_bias=True,
            #                                kernel_initializer=xavier_initializer(),
            #                                trainable=True,
            #                                name='layer_2')
            #
            # tf.logging.warning("layer_2 %s" % conv_output)
            #
            # # size is [batch, channel]
            # conv_output = tf.reduce_mean(conv_output, axis=-2)
            # tf.logging.warning("conv_output %s" % conv_output)
            # conv_output = tf.reshape(conv_output,
            #                          tf.concat([tf.shape(content_embedding)[:2],
            #                                     tf.shape(conv_output)[1:]], axis=0),
            #                          name='fcn_output')
            #
            # tf.logging.info("conv_output %s" % conv_output)
            #
            # return conv_output


def extract_embedding_by_fcn(content_embedding,
                             conv_per_layer,
                             filters,
                             n_layer,
                             is_train,
                             window_size,
                             keep_prob,
                             variable_scope,
                             activation=tf.nn.sigmoid,
                             reuse=True, name=None):
    """
        Extract an embedding for each instance.
    :param content_embedding: [batch_size, n_entities, content_len, word_embedding]
    :param conv_per_layer: For each layer, how many convolution layers will be applied
    :param filters: number of filters
    :param n_layer: How many conv + maxpool layers
    :param is_train: A boolean scalar to control if we do dropout or not
    :param window_size: window size of the conv layer
    :param variable_scope:
    :param reuse:
    :param name:
    :return:
    """

    if len(content_embedding.get_shape()) != 4:
        tf.logging.error("Content embedding must have a rank of 4, get %s" % content_embedding)

    with tf.name_scope(name, 'extract_embedding_by_fcn', [content_embedding, is_train]):
        with tf.variable_scope(variable_scope, reuse=reuse):
            # Reshape the input to [batch, length, channel] so the conv1d has defined shapes
            #  to declare new variables
            conv_output = tf.reshape(content_embedding,
                                     tf.stack([-1,  # batch size
                                               tf.shape(content_embedding)[-2],  # content length
                                               filters]))  # word_embed_size

            # Unlike standard stacked conv structure in which each layer
            # reduces width and depth by half and double the number of channels
            # we believe that most of the information are redundant in the input
            # so we do not need to increase the size of the channels or just slightly
            # increase will be sufficient.
            for layer_id in range(n_layer):
                for conv_layer_id in range(conv_per_layer):
                    conv_output = tf.layers.conv1d(conv_output,
                                                   filters=filters,
                                                   kernel_size=window_size,
                                                   strides=1,
                                                   padding='same',
                                                   activation=activation,
                                                   # change loss to sigmoid to see if we can avoid the nan (without relu the loss will not reduce at all)
                                                   use_bias=True,
                                                   kernel_initializer=xavier_initializer(),
                                                   bias_initializer=xavier_initializer(),
                                                   trainable=True,
                                                   name='layer_%d_conv_%d' % (layer_id, conv_layer_id))
                    # conv_output = tf.check_numerics(conv_output, conv_output.name)
                # add dropout if during training

                conv_output = batch_norm(conv_output,
                                         center=True,
                                         scale=True,
                                         is_training=is_train,
                                         trainable=True,
                                         scope='layer_%d_bn' % layer_id,
                                         decay=0.9)

                # conv_output = tf.layers.batch_normalization(conv_output,
                #                                             training=is_train,
                #                                             trainable=True,
                #                                             name='layer_%d_bn' % layer_id)

                conv_output = tf.cond(is_train,
                                      lambda: tf.nn.dropout(conv_output, keep_prob=keep_prob),
                                      lambda: conv_output)

                if layer_id + 1 == n_layer:
                    # Last layer, reduce the conv_output to [batch_size, n_entities, word_embedding]
                    conv_output = tf.reduce_mean(conv_output, axis=-2)
                else:
                    # reduce content length by half
                    conv_output = tf.layers.max_pooling1d(conv_output,
                                                          pool_size=2,
                                                          strides=2,
                                                          padding='same',
                                                          name='layer_%d_maxpool' % layer_id)
                # conv_output = tf.check_numerics(conv_output, conv_output.name)

            conv_output = tf.reshape(conv_output,
                                     tf.concat([tf.shape(content_embedding)[:2],
                                                tf.shape(conv_output)[1:]], axis=0),
                                     name='fcn_output')

            tf.logging.debug("conv_output %s" % conv_output)

            return conv_output
