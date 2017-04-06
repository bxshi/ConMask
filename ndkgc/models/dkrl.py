import json
import math

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.training as training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ndkgc.ops import get_lookup_table, corrupt_single_relationship, corrupt_single_entity, content_lookup, \
    multiple_content_lookup, normalized_lookup, avg_grads
from ndkgc.utils import count_line, valid_vocab_file, load_list, \
    load_triples, load_pretrained_embedding, load_content


class DKRL(object):
    """ The DKRL Model

    """

    def __init__(self,
                 entity_file,
                 relation_file,
                 vocab_file,
                 pretrain_vocab_file,
                 content_file,
                 train_file,
                 valid_file,
                 test_file,
                 all_triples_file,
                 oov_buckets=10,
                 learning_rate=0.001,
                 margin=1.0,
                 cnn_window=2,
                 structural_embedding_size=100,
                 word_embedding_size=100,
                 feature_map_size=100):

        # TODO: change n_entity to n_train_entity and n_unseen_entity,
        # when doing negative sampling, use n_train_entity as the max_range to avoid unseen entities

        self.entity_file = entity_file
        self.relation_file = relation_file
        self.vocab_file = vocab_file
        self.pretrain_vocab_file = pretrain_vocab_file
        self.content_file = content_file
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.all_triples_file = all_triples_file

        valid_vocab_file(self.vocab_file)

        self.oov_buckets = oov_buckets

        self.learning_rate = learning_rate
        self.margin = margin
        self.cnn_window = cnn_window
        self.structural_embedding_size = structural_embedding_size
        self.word_embedding_size = word_embedding_size
        self.feature_map_size = feature_map_size

        self.n_entity = count_line(self.entity_file)
        self.n_relation = count_line(self.relation_file)
        self.n_vocab = count_line(self.vocab_file)

        self.__initialized = False

        self.entity_table = None
        self.relation_table = None
        self.vocab_table = None

        self.train_matrix = None
        self.test_matrix = None
        self.valid_matrix = None
        self.triple_matrix = None
        self.triple_matrix = None
        self.content_matrix = None

        self.entity_embedding = None
        self.relation_embedding = None
        self.word_embedding = None

        self.global_step = None
        self.lr = None

        self.structural_hrt_w = None
        self.conv_h_structural_rt_w = None
        self.structural_hr_conv_t_w = None
        self.cnn_hrt_w = None

        # Scope for all embeddings
        self.__embedding_scope = None
        with tf.variable_scope('embedding') as scp:
            self.__embedding_scope = scp

        # Scope for all constant variables
        self.__static_variable_scope = None
        with tf.variable_scope('static') as scp:
            self.__static_variable_scope = scp

        # Scope for the actual model
        self.__model_scope = None
        with tf.variable_scope('model') as scp:
            self.__model_scope = scp

        self.__head_scope = None
        with tf.variable_scope('head') as scp:
            self.__head_scope = scp

        self.__tail_scope = None
        with tf.variable_scope('tail') as scp:
            self.__tail_scope = scp

        self.__misc_scope = None
        with tf.variable_scope('misc') as scp:
            self.__misc_scope = scp

    def __conv_layers(self, content, content_len, variable_scope, reuse=True):
        """

        :param content: [batch_size, ?, word_embedding_size]
        :param content_len: [batch_size]
        :return:
        """
        with tf.variable_scope(variable_scope, reuse=reuse):
            if content_len.dtype != tf.float32:
                content_len = tf.cast(content_len, tf.float32)

            conv1_res = tf.layers.conv1d(content,
                                         filters=self.feature_map_size,
                                         kernel_size=2,
                                         padding='valid',
                                         activation=tf.nn.tanh,
                                         kernel_initializer=layers.xavier_initializer(),
                                         name='conv1')
            conv1_maxpool_res = tf.layers.max_pooling1d(conv1_res,
                                                        pool_size=4,
                                                        strides=1,
                                                        name='conv1_maxpool')
            conv2_res = tf.layers.conv1d(conv1_maxpool_res,
                                         filters=self.feature_map_size,
                                         kernel_size=2,
                                         padding='valid',
                                         activation=None,
                                         kernel_initializer=layers.xavier_initializer(),
                                         name='conv2')

        if tf.rank(content_len) == 1:
            content_len = tf.reshape(content_len, [-1, 1])

        # [batch_size, feature_map_size]
        conv2_meanpool_res = tf.truediv(tf.reduce_sum(conv2_res, axis=1),
                                        tf.expand_dims(content_len, axis=1))

        return conv2_meanpool_res

    def load_static_variables(self, sess):
        entity_dict = load_list(self.entity_file)
        relation_dict = load_list(self.relation_file)

        train_triples = load_triples(self.train_file,
                                     entity_dict,
                                     relation_dict)
        if not self.__initialized:
            self.__initialize_model()

        self.train_matrix.load(np.asarray(train_triples), sess)
        del train_triples

        if self.valid_matrix is not None:
            valid_triples = load_triples(self.valid_file,
                                         entity_dict,
                                         relation_dict)
            self.valid_matrix.load(np.asarray(valid_triples), sess)
            del valid_triples

        if self.test_matrix is not None:
            test_triples = load_triples(self.test_file,
                                        entity_dict,
                                        relation_dict)
            self.test_matrix.load(np.asarray(test_triples), sess)
            del test_triples

        all_triples = load_triples(self.all_triples_file,
                                   entity_dict,
                                   relation_dict)

        self.triple_matrix.load(np.asarray(all_triples), sess)
        del all_triples

        vocab = load_list(self.vocab_file)

        self.word_embedding.load(load_pretrained_embedding(self.pretrain_vocab_file,
                                                           vocab,
                                                           self.word_embedding_size,
                                                           self.oov_buckets), sess)
        del vocab

        self.content_matrix.load(load_content(self.content_file, entity_dict), sess)

    def dist(self, h, r, t):
        return tf.reduce_sum(tf.abs(h + r - t), axis=-1)

    def ranking_loss(self, positive_dist, negative_dist):
        return tf.reduce_mean(tf.maximum(self.margin + positive_dist - negative_dist, 0.))

    def inference(self, triples, head_content_ids, head_content_len,
                  tail_content_ids, tail_content_len, variable_scope, reuse=True):
        with tf.variable_scope(variable_scope, reuse=reuse, values=[self.entity_embedding,
                                                                    self.relation_embedding,
                                                                    self.word_embedding,
                                                                    triples]):
            print("inference triples shape", triples.get_shape())

            with tf.device('/cpu:0'):
                heads, rels, tails = tf.unstack(triples, axis=1, name='unstack_hrt')

                heads_embed = normalized_lookup(self.entity_embedding,
                                                heads,
                                                name='head_embedding_lookup')
                rels_embed = normalized_lookup(self.relation_embedding,
                                               rels,
                                               name='relation_embedding_lookup')
                tails_embed = normalized_lookup(self.entity_embedding,
                                                tails,
                                                name='tail_embedding_lookup')
                # [batch_size, ?, word_embedding_size]
                heads_content = normalized_lookup(self.word_embedding, head_content_ids,
                                                  name='head_content_lookup')
                # [batch_size, ?, word_embedding_size]
                tails_content = normalized_lookup(self.word_embedding, tail_content_ids,
                                                  name='tail_content_lookup')

            # First, |h + r - t| for h,r,t all using structural embeddings
            structural_hrt_score = self.dist(heads_embed, rels_embed, tails_embed)

            # Convolution Layers

            with tf.variable_scope('head_conv', reuse=reuse):
                heads_cnn_embed = self.__conv_layers(heads_content, head_content_len,
                                                     self.__head_scope, reuse=reuse)

            with tf.variable_scope('tail_conv', reuse=reuse):
                tails_cnn_embed = self.__conv_layers(tails_content, tail_content_len,
                                                     self.__tail_scope, reuse=reuse)

            # |h_{cnn} + r - t|

            conv_h_structural_rt_score = self.dist(heads_cnn_embed, rels_embed, tails_embed)

            # |h + r - t_{cnn}|
            structural_hr_conv_t_score = self.dist(heads_embed, rels_embed, tails_cnn_embed)

            # |h_{cnn} + r - t_{cnn}|
            cnn_hrt_score = self.dist(heads_cnn_embed, rels_embed, tails_cnn_embed)

            # return structural_hrt_score * self.structural_hrt_w + \
            #        conv_h_structural_rt_score * self.conv_h_structural_rt_w + \
            #        structural_hr_conv_t_score * self.structural_hr_conv_t_w + \
            #        cnn_hrt_score * self.cnn_hrt_w

            return structural_hrt_score + conv_h_structural_rt_score + structural_hr_conv_t_score + cnn_hrt_score

    def __initialize_model(self):
        if self.__initialized:
            return
        self.__initialized = True

        print("Run __initialize_model()")

        # Lookup tables
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.__static_variable_scope):
                self.entity_table = get_lookup_table(self.entity_file,
                                                     oov_buckets=0,
                                                     size=self.n_entity,
                                                     name='entity_lookup_table')
                self.relation_table = get_lookup_table(self.relation_file,
                                                       oov_buckets=0,
                                                       size=self.n_relation,
                                                       name='relation_lookup_table')

                self.vocab_table = get_lookup_table(self.vocab_file,
                                                    oov_buckets=self.oov_buckets,
                                                    size=self.n_vocab,
                                                    name='vocab_lookup_table')

                self.content_matrix = tf.get_variable("content_matrix",
                                                      dtype=tf.string,
                                                      initializer=[""] * self.n_entity,
                                                      trainable=False,
                                                      collections=['static_variables'])

                self.train_matrix = tf.get_variable("train_triple",
                                                    [count_line(self.train_file), 3],
                                                    dtype=tf.int32,
                                                    trainable=False,
                                                    collections=['static_variables'])
                if self.test_file:
                    self.test_matrix = tf.get_variable("test_triple",
                                                       [count_line(self.test_file), 3],
                                                       dtype=tf.int32,
                                                       trainable=False,
                                                       collections=['static_variables'])
                if self.valid_file:
                    self.valid_matrix = tf.get_variable("valid_triple",
                                                        [count_line(self.valid_file), 3],
                                                        dtype=tf.int32,
                                                        trainable=False,
                                                        collections=['static_variables'])

                self.triple_matrix = tf.get_variable("all_triples",
                                                     [count_line(self.all_triples_file), 3],
                                                     dtype=tf.int32,
                                                     trainable=False,
                                                     collections=['static_variables'])

            # Embeddings
            with tf.variable_scope(self.__embedding_scope):
                self.entity_embedding = tf.get_variable("entity_embedding",
                                                        [self.n_entity, self.structural_embedding_size],
                                                        dtype=tf.float32,
                                                        initializer=layers.xavier_initializer(),
                                                        trainable=True)

                self.relation_embedding = tf.get_variable("relation_embedding",
                                                          [self.n_relation, self.structural_embedding_size],
                                                          dtype=tf.float32,
                                                          initializer=layers.xavier_initializer(),
                                                          trainable=True)

                self.word_embedding = tf.get_variable("word_embedding",
                                                      [self.n_vocab + self.oov_buckets, self.word_embedding_size],
                                                      dtype=tf.float32,
                                                      initializer=layers.xavier_initializer(),
                                                      trainable=True)

            with tf.variable_scope(self.__misc_scope):
                self.global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='global_step')
                self.lr = tf.Variable(self.learning_rate, dtype=tf.float32, trainable=False, name='lr')

                # Debug variables
                self.head_corrupted = tf.Variable(0, dtype=tf.int32, trainable=False, name='head_corrupted')
                self.tail_corrupted = tf.Variable(0, dtype=tf.int32, trainable=False, name='tail_corrupted')

                # metrics
                self.head_mean_rank = tf.Variable(0, dtype=tf.float64, trainable=False, name='head_mean_rank')
                self.tail_mean_rank = tf.Variable(0, dtype=tf.float64, trainable=False, name='tail_mean_rank')
                self.filtered_head_mean_rank = tf.Variable(0, dtype=tf.float64, trainable=False,
                                                           name='head_filtered_mean_rank')
                self.filtered_tail_mean_rank = tf.Variable(0, dtype=tf.float64, trainable=False,
                                                           name='tail_filtered_mean_rank')
                self.head_hits = tf.Variable(0, dtype=tf.float64, trainable=False, name='head_hits')
                self.tail_hits = tf.Variable(0, dtype=tf.float64, trainable=False, name='tail_hits')
                self.filtered_head_hits = tf.Variable(0, dtype=tf.float64, trainable=False, name='head_filtered_hits')
                self.filtered_tail_hits = tf.Variable(0, dtype=tf.float64, trainable=False, name='tail_filtered_hits')
                self.eval_instances = tf.Variable(0, dtype=tf.int32, trainable=False, name='eval_instance')

            with tf.variable_scope(self.__model_scope):

                ph_triples = tf.placeholder(tf.int32, [None, 3], name='ph_triples')
                ph_content = tf.placeholder(tf.int32, [None, None], name='ph_content')
                ph_content_len = tf.placeholder(tf.int32, [None], name='ph_content_len')
                self.inference(triples=ph_triples,
                               head_content_ids=ph_content,
                               head_content_len=ph_content_len,
                               tail_content_ids=ph_content,
                               tail_content_len=ph_content_len,
                               variable_scope=self.__model_scope, reuse=False)

    def train_op(self, num_epochs=10, batch_size=200):

        if not self.__initialized:
            self.__initialize_model()
        # Build input pipeline
        with tf.name_scope('input_pipeline', [self.train_matrix,
                                              self.triple_matrix,
                                              self.content_matrix, self.vocab_table]):
            with tf.device('/cpu:0'):
                input_triples = tf.train.limit_epochs(
                    tf.random_shuffle(self.train_matrix, name='shuffled_input_triples'),
                    num_epochs=num_epochs,
                    name='train_triples_limited')
                single_triple = tf.train.shuffle_batch([input_triples],
                                                       batch_size=1,
                                                       capacity=min(batch_size * 40, self.train_matrix.get_shape()[0]),
                                                       min_after_dequeue=min(batch_size * 15,
                                                                             self.train_matrix.get_shape()[0]),
                                                       enqueue_many=True,
                                                       allow_smaller_final_batch=True,
                                                       name='train_shuffle_batch')
                # make sure it is a 1-d vector
                single_triple = tf.reshape(single_triple, [3])

                relation_corrupted_triple = corrupt_single_relationship(single_triple,
                                                                        self.triple_matrix,
                                                                        self.n_relation - 1)

                entity_corrupted_triple = corrupt_single_entity(single_triple,
                                                                self.train_matrix,
                                                                self.n_entity - 1,
                                                                debug_head_corrupted=self.head_corrupted,
                                                                debug_tail_corrupted=self.tail_corrupted)

                head_content_ids = content_lookup(self.content_matrix,
                                                  self.vocab_table,
                                                  single_triple[0],
                                                  name='h_content_lookup')
                head_content_len = tf.cast(tf.shape(head_content_ids)[0], tf.int32)

                tail_content_ids = content_lookup(self.content_matrix,
                                                  self.vocab_table,
                                                  single_triple[2],
                                                  name='t_content_lookup')
                tail_content_len = tf.cast(tf.shape(tail_content_ids)[0], tf.int32)

                corrupted_head_content_id = content_lookup(self.content_matrix,
                                                           self.vocab_table,
                                                           entity_corrupted_triple[0],
                                                           name='corrupted_h_content_lookup')
                corrupted_head_content_len = tf.cast(tf.shape(corrupted_head_content_id)[0], tf.int32)

                corrupted_tail_content_id = content_lookup(self.content_matrix,
                                                           self.vocab_table,
                                                           entity_corrupted_triple[2],
                                                           name='corrupted_t_content_lookup')
                corrupted_tail_content_len = tf.cast(tf.shape(corrupted_tail_content_id)[0], tf.int32)

                # Get content information for
                single_instance_max_len = tf.maximum(head_content_len,
                                                     tf.maximum(tail_content_len,
                                                                tf.maximum(corrupted_head_content_len,
                                                                           corrupted_tail_content_len)))

                batch_input_tensors = [single_triple,
                                       entity_corrupted_triple,
                                       relation_corrupted_triple,
                                       head_content_ids, head_content_len,
                                       tail_content_ids, tail_content_len,
                                       corrupted_head_content_id, corrupted_head_content_len,
                                       corrupted_tail_content_id, corrupted_tail_content_len]

                # TODO: Change to tf.contrib.training.bucket_by_sequence_length
                # We need to put all *_content_ids into a matrix otherwise it will throw an error
                # because it want to pack all elements together.
                # _, input_queue = training.bucket_by_sequence_length(single_instance_max_len,
                #                                                     [batch_input_tensors],
                #                                                     batch_size=batch_size * 4,
                #                                                     bucket_boundaries=[20, 40, 60, 80, 120],
                #                                                     capacity=batch_size * 4 * 10,
                #                                                     # shapes=[[3], [3], [3],
                #                                                     #         [None], (),
                #                                                     #         [None], (),
                #                                                     #         [None], (),
                #                                                     #         [None], ()],
                #                                                     dynamic_pad=True,
                #                                                     allow_smaller_final_batch=True,
                #                                                     name="input_bucketed_queue")
                #
                # print("INPUT_QUEUE", input_queue)

                input_queue = tf.train.batch(batch_input_tensors,
                                             batch_size=batch_size * 4,
                                             num_threads=4,
                                             capacity=min(batch_size * 40, self.train_matrix.get_shape()[0]),
                                             enqueue_many=False,
                                             shapes=[[3], [3], [3],
                                                     [None], (),
                                                     [None], (),
                                                     [None], (),
                                                     [None], ()],
                                             dynamic_pad=True,
                                             allow_smaller_final_batch=True,
                                             name='input_queue')

        with tf.name_scope('train', values=input_queue):
            # Inputs of the actual models

            with tf.device('/cpu:0'):
                triple_batch, entity_corrupted_triple_batch, \
                relation_corrupted_triple_batch, \
                head_content_ids_batch, head_content_len_batch, \
                tail_content_ids_batch, tail_content_len_batch, \
                corrupted_head_content_id_batch, corrupted_head_content_len_batch, \
                corrupted_tail_content_id_batch, corrupted_tail_content_len_batch = [tf.split(x, 4) for x in
                                                                                     input_queue]

                optimizer = tf.train.AdamOptimizer(self.lr)

                tower_grads = list()
                losses = list()

            for gpu_id in range(4):
                with tf.device('/gpu:%d' % gpu_id):
                    triple_score = self.inference(triples=triple_batch[gpu_id],
                                                  head_content_ids=head_content_ids_batch[gpu_id],
                                                  head_content_len=head_content_len_batch[gpu_id],
                                                  tail_content_ids=tail_content_ids_batch[gpu_id],
                                                  tail_content_len=tail_content_len_batch[gpu_id],
                                                  variable_scope=self.__model_scope)

                    entity_corrupted_triple_score = self.inference(triples=entity_corrupted_triple_batch[gpu_id],
                                                                   head_content_ids=corrupted_head_content_id_batch[
                                                                       gpu_id],
                                                                   head_content_len=corrupted_head_content_len_batch[
                                                                       gpu_id],
                                                                   tail_content_ids=corrupted_tail_content_id_batch[
                                                                       gpu_id],
                                                                   tail_content_len=corrupted_tail_content_len_batch[
                                                                       gpu_id],
                                                                   variable_scope=self.__model_scope)

                    relation_corrupted_triple_score = self.inference(triples=relation_corrupted_triple_batch[gpu_id],
                                                                     head_content_ids=head_content_ids_batch[gpu_id],
                                                                     head_content_len=head_content_len_batch[gpu_id],
                                                                     tail_content_ids=tail_content_ids_batch[gpu_id],
                                                                     tail_content_len=tail_content_len_batch[gpu_id],
                                                                     variable_scope=self.__model_scope)

                    loss = self.ranking_loss(triple_score,
                                             entity_corrupted_triple_score) + self.ranking_loss(triple_score,
                                                                                                relation_corrupted_triple_score)

                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    losses.append(loss)

            with tf.device('/cpu:0'):
                grads = avg_grads(tower_grads)
                train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
                loss_op = tf.reduce_mean(tf.stack(losses))
            return train_op, loss_op

    def head_conv_helper(self, x):
        content_ids, content_len = multiple_content_lookup(self.content_matrix,
                                                           self.vocab_table,
                                                           x)
        content_embedding = normalized_lookup(self.word_embedding, content_ids)
        return content_ids, content_len, content_embedding, self.__conv_layers(content_embedding, content_len,
                                                                               self.__head_scope)

    def tail_conv_helper(self, x):
        content_ids, content_len = multiple_content_lookup(self.content_matrix,
                                                           self.vocab_table,
                                                           x)
        content_embedding = normalized_lookup(self.word_embedding, content_ids)
        return content_ids, content_len, content_embedding, self.__conv_layers(content_embedding, content_len,
                                                                               self.__tail_scope)

    def _conv_helper(self, x, scope):
        content_ids, content_len = multiple_content_lookup(self.content_matrix,
                                                           self.vocab_table,
                                                           x)
        content_embedding = normalized_lookup(self.word_embedding, content_ids)
        return self.__conv_layers(content_embedding, content_len, scope)

    def eval(self, eval_type, batch_size=100, precompute_split=10):

        if not self.__initialized:
            self.__initialize_model()

        if eval_type == 'train':
            eval_matrix = self.train_matrix
        elif eval_type == 'valid':
            eval_matrix = self.valid_matrix
        elif eval_type == 'test':
            eval_matrix = self.test_matrix
        else:
            raise ValueError("No %s set for evaluation!")

        with tf.variable_scope('eval_precompute'):
            # Create variables to store precomputed CNN results for head entities and tail entities
            head_conv_embed = tf.get_variable("head_conv_embed",
                                              [self.n_entity, self.feature_map_size],
                                              dtype=tf.float32,
                                              trainable=False,
                                              initializer=tf.zeros_initializer(),
                                              collections=['static_variables', tf.GraphKeys.GLOBAL_VARIABLES])
            tail_conv_embed = tf.get_variable("tail_conv_embed",
                                              [self.n_entity, self.feature_map_size],
                                              dtype=tf.float32,
                                              trainable=False,
                                              initializer=tf.zeros_initializer(),
                                              collections=['static_variables', tf.GraphKeys.GLOBAL_VARIABLES])

            _precomptue_batch_size = int(math.floor(self.n_entity / precompute_split))
            precompute_batch_size = [_precomptue_batch_size] * (precompute_split - 1) + [
                self.n_entity - _precomptue_batch_size * (precompute_split - 1)]
            # precompute_batch_size = [1] * 2
            assert sum(precompute_batch_size) == self.n_entity

            # Split entities into precompute_split tensors
            entity_batches = tf.split(tf.range(0, self.n_entity), precompute_batch_size)

            # You need to run all ops in precompute_*_ops, concatenate them together and assign
            # to *_conv_embed before running other ops
            precompute_head_conv_ops = list()
            precompute_tail_conv_ops = list()
            for entity_batch, size in zip(entity_batches, precompute_batch_size):
                precompute_head_conv_ops.append(self._conv_helper(entity_batch, self.__head_scope))
                precompute_tail_conv_ops.append(self._conv_helper(entity_batch, self.__tail_scope))

        with tf.name_scope('eval_input_pipeline', values=[eval_matrix,
                                                          self.triple_matrix,
                                                          self.content_matrix, self.vocab_table]):
            input_triple_matrix = tf.train.limit_epochs(eval_matrix, num_epochs=1,
                                                        name='eval_triples_limited')

            # [batch_size, 3]
            triples = tf.train.batch([input_triple_matrix],
                                     batch_size=batch_size,
                                     capacity=batch_size * 10,
                                     shapes=[[3]],
                                     enqueue_many=True,
                                     allow_smaller_final_batch=True,
                                     name='input_batch')

        with tf.name_scope('eval'):
            # First generate convolution embedding for
            # all heads and all tails so we could reuse them

            # [1, n_entity, feature_dim] use this to do math with partial_*_*
            # This equals to getting the result using normalized_lookup
            _head_conv = tf.expand_dims(head_conv_embed, axis=0) / float(self.feature_map_size)
            _tail_conv = tf.expand_dims(tail_conv_embed, axis=0) / float(self.feature_map_size)
            _ent_embed = tf.expand_dims(self.entity_embedding, axis=0) / float(self.word_embedding_size)

            # three 1-D [batch_size] vectors
            heads, rels, tails = tf.unstack(triples, axis=1)

            # [batch_size, 1, feature_dim]
            partial_head_conv = tf.expand_dims(normalized_lookup(head_conv_embed, heads), axis=1)
            partial_tail_conv = tf.expand_dims(normalized_lookup(tail_conv_embed, tails), axis=1)

            partial_head_embed = tf.expand_dims(normalized_lookup(self.entity_embedding, heads), axis=1)
            partial_tail_embed = tf.expand_dims(normalized_lookup(self.entity_embedding, tails), axis=1)

            partial_rel_embed = tf.expand_dims(normalized_lookup(self.relation_embedding, rels), axis=1)

            # pred_head_scores
            pred_head_dd_score = self.dist(_head_conv, partial_rel_embed, partial_tail_conv)
            pred_head_ds_score = self.dist(_head_conv, partial_rel_embed, partial_tail_embed)
            pred_head_sd_score = self.dist(_ent_embed, partial_rel_embed, partial_tail_conv)
            pred_head_ss_score = self.dist(_ent_embed, partial_rel_embed, partial_tail_embed)

            # pred_head_scores = pred_head_dd_score * self.cnn_hrt_w + \
            #                    (pred_head_sd_score + pred_head_ds_score) * self.conv_h_structural_rt_w + \
            #                    pred_head_ss_score * self.structural_hrt_w

            pred_head_scores = pred_head_dd_score + pred_head_sd_score + pred_head_ds_score + pred_head_ss_score

            # pred_tail_scores
            pred_tail_dd_score = self.dist(partial_head_conv, partial_rel_embed, _tail_conv)
            pred_tail_ds_score = self.dist(partial_head_conv, partial_rel_embed, _ent_embed)
            pred_tail_sd_score = self.dist(partial_head_embed, partial_rel_embed, _tail_conv)
            pred_tail_ss_score = self.dist(partial_head_embed, partial_rel_embed, _ent_embed)

            # pred_tail_scores = pred_tail_dd_score * self.cnn_hrt_w + \
            #                    (pred_tail_sd_score + pred_tail_ds_score) * self.conv_h_structural_rt_w + \
            #                    pred_tail_ss_score * self.structural_hrt_w

            pred_tail_scores = pred_tail_dd_score + pred_tail_sd_score + pred_tail_ds_score + pred_tail_ss_score

            # Calculate metrics
            def _matched_ents_helper(hrt, hrt_matrix):
                h, r, t = tf.unstack(hrt)
                print("matched_ents_helper hrt shape", hrt.get_shape())
                with tf.name_scope("matched_ent_helper"):
                    rel_mask = tf.equal(hrt_matrix[:, 1], r)
                    rel_matched_triples = tf.boolean_mask(hrt_matrix, rel_mask)

                    tail_mask = tf.equal(rel_matched_triples[:, 0], h)
                    head_mask = tf.equal(rel_matched_triples[:, 2], t)

                    print("rel_matched_triples shape", rel_matched_triples.get_shape())
                    print("tail_mask shape", tail_mask.get_shape())
                    print("head_mask shape", head_mask.get_shape())

                    # tails, _ = tf.unique(
                    #     tf.boolean_mask(rel_matched_triples[:, 2], tail_mask))
                    tails, _ = tf.unique(tf.boolean_mask(rel_matched_triples[:, 2], tail_mask))
                    # tails = tf.boolean_mask(rel_matched_triples[:, 2], tail_mask)

                    # heads, _ = tf.unique(
                    #     tf.boolean_mask(rel_matched_triples[:, 0], head_mask))
                    heads, _ = tf.unique(tf.boolean_mask(rel_matched_triples[:, 0], head_mask))
                    # heads = tf.boolean_mask(rel_matched_triples[:, 0], head_mask)

                    print("_matched_ents_helper heads shape", heads.get_shape())
                    print("_matched_ents_helper tails shape", tails.get_shape())

                    return heads, tails

            # self.debug_res = _matched_ents_helper(triple_batch[0], eval_matrix)
            # self.debug_res = tf.map_fn(lambda x: _matched_ents_helper(x, eval_matrix),
            #                            triple_batch,
            #                            dtype=tf.int32, back_prop=False)

            def metric_helper(x):
                _triple, _pred_head_score, _pred_tail_score = x
                print("metric helper triple ", _triple.get_shape())
                print("metric helper pred_head_score", _pred_head_score.get_shape())
                print("metric helper pred_tail_score", _pred_tail_score.get_shape())

                # true_heads, true_tails = _triple[:1], _triple[2:]
                true_heads, true_tails = _matched_ents_helper(_triple, self.triple_matrix)

                # eval_heads, eval_tails = _triple[:1], _triple[2:]
                eval_heads, eval_tails = _matched_ents_helper(_triple, eval_matrix)

                # e1 = tf.assert_less(true_heads, self.n_entity)
                # e2 = tf.assert_less(true_tails, self.n_entity)
                # e3 = tf.assert_less(eval_heads, self.n_entity)
                # e4 = tf.assert_less(eval_tails, self.n_entity)

                # with tf.control_dependencies([e1, e2, e3, e4]):
                #     res = tf.cast(tf.stack([true_heads[0],
                #                          true_tails[0],
                #                          eval_heads[0],
                #                          eval_tails[0],
                #                          true_heads[0],
                #                          true_tails[0],
                #                          eval_heads[0],
                #                          eval_tails[0],], axis=0), tf.float64)
                #     return res

                # true_head_score_mask = tf.sparse_to_dense(true_heads, [self.n_entity],
                #                                           tf.ones_like(true_heads, dtype=tf.float32) * 1e10,
                #                                           name='true_head_score_mask')

                # true_tail_score_mask = tf.sparse_to_dense(true_tails, [self.n_entity],
                #                                           tf.ones_like(true_tails, dtype=tf.float32) * 1e10,
                #                                           name='true_head_score_mask')

                # print("true_head_score_mask shape", true_head_score_mask.get_shape())
                # print("true_tail_score_mask shape", true_tail_score_mask.get_shape())

                true_head_score_mask = tf.SparseTensor(indices=tf.cast(tf.reshape(true_heads, [-1, 1]), tf.int64),
                                                       values=tf.ones_like(true_heads, dtype=tf.float32) * 1e10,
                                                       dense_shape=[self.n_entity])

                true_tail_score_mask = tf.SparseTensor(indices=tf.cast(tf.reshape(true_tails, [-1, 1]), tf.int64),
                                                       values=tf.ones_like(true_tails, dtype=tf.float32) * 1e10,
                                                       dense_shape=[self.n_entity])

                # [?] =? [?, 1]
                eval_head_scores = tf.expand_dims(tf.nn.embedding_lookup(_pred_head_score, eval_heads), axis=1)
                eval_tail_scores = tf.expand_dims(tf.nn.embedding_lookup(_pred_tail_score, eval_tails), axis=1)

                print("eval_head_scores shape", eval_head_scores.get_shape())
                print("eval_tail_scores shape", eval_tail_scores.get_shape())

                # [n_entity] => [1, n_entity]
                masked_head_score = tf.expand_dims(tf.sparse_add(_pred_head_score, true_head_score_mask), axis=0)
                # masked_head_score = tf.expand_dims(_pred_head_score + true_head_score_mask, axis=0)
                masked_tail_score = tf.expand_dims(tf.sparse_add(_pred_tail_score, true_tail_score_mask), axis=0)
                # masked_tail_score = tf.expand_dims(_pred_tail_score + true_tail_score_mask, axis=0)

                print("masked_head_score shape", masked_head_score.get_shape())
                print("masked_tail_score shape", masked_tail_score.get_shape())

                # Reshape pred_*_score to [1, batch_size, n_entity]
                _pred_head_score = tf.expand_dims(_pred_head_score, axis=0)
                _pred_tail_score = tf.expand_dims(_pred_tail_score, axis=0)

                print("_pred_head_score shape", _pred_head_score.get_shape())
                print("_pred_tail_score shape", _pred_tail_score.get_shape())

                # [?]
                head_rank = tf.reduce_sum(tf.cast(tf.less(_pred_head_score, eval_head_scores), tf.float64)) + 1.
                filtered_head_rank = tf.reduce_sum(
                    tf.cast(tf.less(masked_head_score, eval_head_scores), tf.float64)) + 1.
                tail_rank = tf.reduce_sum(tf.cast(tf.less(_pred_tail_score, eval_tail_scores), tf.float64)) + 1.
                filtered_tail_rank = tf.reduce_sum(
                    tf.cast(tf.less(masked_tail_score, eval_tail_scores), tf.float64)) + 1.

                head_hit = tf.reduce_sum(tf.cast(tf.less_equal(head_rank, 10), tf.float64))
                filtered_head_hit = tf.reduce_sum(tf.cast(tf.less(filtered_head_rank, 10), tf.float64))

                tail_hit = tf.reduce_sum(tf.cast(tf.less_equal(tail_rank, 10), tf.float64))
                filtered_tail_hit = tf.reduce_sum(tf.cast(tf.less(filtered_tail_rank, 10), tf.float64))

                return tf.stack([tf.reshape(head_rank, ()),
                                 tf.reshape(filtered_head_rank, ()),
                                 head_hit,
                                 filtered_head_hit,
                                 tf.reshape(tail_rank, ()),
                                 tf.reshape(filtered_tail_rank, ()),
                                 tail_hit,
                                 filtered_tail_hit], axis=0)

            print("PRED_HEAD_SCORES SHAPE", pred_head_scores.get_shape())
            print("PRED_TAIL_SCORES SHAPE", pred_tail_scores.get_shape())

            _head_rank, _filtered_head_rank, \
            _head_hit, _filtered_head_hit, \
            _tail_rank, _filtered_tail_rank, \
            _tail_hit, _filtered_tail_hit = [
                tf.reduce_sum(x) for x in tf.split(tf.map_fn(metric_helper,
                                                             [triples,
                                                              pred_head_scores,
                                                              pred_tail_scores],
                                                             dtype=tf.float64,
                                                             back_prop=False,
                                                             swap_memory=True,
                                                             name='eval_map_fn'), 8,
                                                   axis=1, name='eval_metric_split')]

            eval_op = tf.group(self.head_mean_rank.assign_add(_head_rank),
                               self.filtered_head_mean_rank.assign_add(_filtered_head_rank),
                               self.head_hits.assign_add(_head_hit),
                               self.filtered_head_hits.assign_add(_filtered_head_hit),
                               self.tail_mean_rank.assign_add(_tail_rank),
                               self.filtered_tail_mean_rank.assign_add(_filtered_tail_rank),
                               self.tail_hits.assign_add(_tail_hit),
                               self.filtered_tail_hits.assign_add(_filtered_tail_hit),
                               self.eval_instances.assign_add(tf.shape(triples)[0]))

            reset_op = tf.group(self.head_mean_rank.assign(0.),
                                self.filtered_head_mean_rank.assign(0.),
                                self.head_hits.assign(0.),
                                self.filtered_head_hits.assign(0.),
                                self.tail_mean_rank.assign(0.),
                                self.filtered_tail_mean_rank.assign(0.),
                                self.tail_hits.assign(0.),
                                self.filtered_tail_hits.assign(0.),
                                self.eval_instances.assign(0))

            eval_instance_flt = tf.cast(self.eval_instances, tf.float64)
            metric_op = {
                'head': {
                    'mean_rank': tf.truediv(self.head_mean_rank, eval_instance_flt),
                    'filtered_mean_rank': tf.truediv(self.filtered_head_mean_rank, eval_instance_flt),
                    'hits': tf.truediv(self.head_hits, eval_instance_flt),
                    'filtered_hits': tf.truediv(self.filtered_head_hits, eval_instance_flt)
                },
                'tail': {
                    'mean_rank': tf.truediv(self.tail_mean_rank, eval_instance_flt),
                    'filtered_mean_rank': tf.truediv(self.filtered_tail_mean_rank, eval_instance_flt),
                    'hits': tf.truediv(self.tail_hits, eval_instance_flt),
                    'filtered_hits': tf.truediv(self.filtered_tail_hits, eval_instance_flt)
                },
                'n_instance': self.eval_instances
            }

            # eval_op = tf.no_op()
            # reset_op = tf.no_op()
            # metric_op = tf.no_op()

            return precompute_head_conv_ops, head_conv_embed, precompute_tail_conv_ops, tail_conv_embed, eval_op, reset_op, metric_op


def main(_):
    import os

    DATA_DIR = './data/fb15k/'
    model = DKRL(
        entity_file=os.path.join(DATA_DIR, 'entities.txt'),
        relation_file=os.path.join(DATA_DIR, 'relations.txt'),
        vocab_file=os.path.join(DATA_DIR, 'vocab.txt'),
        pretrain_vocab_file=os.path.join(DATA_DIR, 'glove.6B.100d.txt'),
        content_file=os.path.join(DATA_DIR, 'descriptions.txt'),
        train_file=os.path.join(DATA_DIR, 'train.txt'),
        valid_file=os.path.join(DATA_DIR, 'valid.txt'),
        test_file=os.path.join(DATA_DIR, 'test.txt'),
        all_triples_file=os.path.join(DATA_DIR, 'all_triples.txt')
    )

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    train_op, loss_op = model.train_op(num_epochs=10, batch_size=1024)
    head_precompute_ops, head_conv_embed, tail_precompute_ops, tail_conv_embed, \
    eval_op, reset_op, metric_op = model.eval(eval_type='test', batch_size=200)

    saver = tf.train.Saver(max_to_keep=3)

    def train():
        with tf.Session(config=config) as sess:
            model.load_static_variables(sess)

            sess.run([tf.tables_initializer(),
                      tf.global_variables_initializer(),
                      tf.local_variables_initializer()])

            print("All variables initialized.")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint('./checkpoint/'))
            except tf.errors.NotFoundError:
                tf.logging.error("You may have changed your model and there "
                                 "are new variables that can not be load from previous snapshot. "
                                 "We will keep running but be aware that parts of your model are"
                                 " RANDOM MATRICES!")

            try:
                cnt = 0
                while not coord.should_stop():
                    cnt += 1
                    if cnt % 10 == 0:
                        _, loss, global_step = sess.run([train_op, loss_op, model.global_step])
                        print("GSTEP:_%d_LOSS:_%.4f" % (global_step, loss), end='\r')
                    else:
                        sess.run(train_op)
            except tf.errors.OutOfRangeError:
                print("training done")
            finally:
                coord.request_stop()
            coord.join(threads)

            saver.save(sess, "./checkpoint/model.ckpt", global_step=model.global_step)
            tf.logging.info("Model saved")

    def eval():
        with tf.Session(config=config) as sess:

            sess.run([tf.tables_initializer(),
                      tf.global_variables_initializer(),
                      tf.local_variables_initializer()])

            model.load_static_variables(sess)

            print("All variables initialized.")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint('./checkpoint/'))
            except tf.errors.NotFoundError:
                tf.logging.error("You may have changed your model and there "
                                 "are new variables that can not be load from previous snapshot. "
                                 "We will keep running but be aware that parts of your model are"
                                 " RANDOM MARTICES!")

            # print("entity_embedding", sess.run(model.entity_embedding))
            # print("relation_embedding", sess.run(model.relation_embedding))

            try:
                head_conv = list()
                tail_conv = list()

                # print("head_conv_embed", sess.run(head_conv_embed))
                # print("tail_conv_embed", sess.run(tail_conv_embed))

                for c, (head_precompute, tail_precompute) in enumerate(zip(head_precompute_ops, tail_precompute_ops)):
                    head, tail = sess.run([head_precompute, tail_precompute])
                    print("precomputing CNN embeddings %d/%d" % (c, len(tail_precompute_ops)), end='\r')
                    head_conv.append(head)
                    tail_conv.append(tail)
                head_conv_embed.load(np.concatenate(head_conv, axis=0), sess)
                tail_conv_embed.load(np.concatenate(tail_conv, axis=0), sess)

                del head_conv, tail_conv

                # print(sess.run(head_conv_embed))
                # print(sess.run(tail_conv_embed))

                # print(model.triple_matrix)
                # print(model.train_matrix)
                # print(model.valid_matrix)
                # print(model.test_matrix)

                # print("max ", sess.run(tf.reduce_max(model.triple_matrix[:, 0])))
                # print("max ", sess.run(tf.reduce_max(model.triple_matrix[:, 1])))
                # print("max ", sess.run(tf.reduce_max(model.triple_matrix[:, 2])))
                cnt = 0

                sess.run(reset_op)

                while not coord.should_stop():
                    cnt += 1
                    if cnt * 200 % 10000 == 0:
                        _, metric = sess.run([eval_op, metric_op])
                        js_str = json.loads(str(metric).replace("'", '"'))
                        print(json.dumps(js_str, sort_keys=True, indent=2))
                    else:
                        sess.run(eval_op)
            except tf.errors.OutOfRangeError:
                print("training done")
            finally:
                coord.request_stop()
                js_str = json.loads(str(sess.run(metric_op)).replace("'", '"'))
                print(json.dumps(js_str, sort_keys=True, indent=2))
            coord.join(threads)

    while True:
        train()
        eval()


if __name__ == '__main__':
    tf.app.run()
