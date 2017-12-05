import csv
from intbitset import intbitset

import tensorflow.contrib.layers as layers

from ndkgc.ops import *
from ndkgc.utils import *

"""
    This model only averaging the content information, not including the title
"""

tf.app.flags.DEFINE_boolean('eval', False, 'Run evaluation')
tf.app.flags.DEFINE_boolean('force_eval', False, 'Run evaluation during training instead of validation')
tf.app.flags.DEFINE_integer('max_content', 256, 'Max content length')
tf.app.flags.DEFINE_integer('batch', 200, 'batch_size')
FLAGS = tf.app.flags.FLAGS


class ContentAveragingModel(object):
    PAD = '__PAD__'
    PAD_const = tf.constant(PAD, name='pad')

    TRAIN_SUMMARY = 'train_summary'
    TRAIN_SUMMARY_SLOW = 'train_summary_slow'

    EVAL_SUMMARY = 'eval_summary'

    def __init__(self, **kwargs):
        """ Initialize all parameters and variable scopes for a content-based model.

        If you have new parameters you want to add, inherit from this class
        and add yours in the child class.

        :param kwargs:
        """
        # entity string name per line, no space or tab
        self.entity_file = kwargs['entity_file']
        self.n_entity = count_line(self.entity_file)
        # relation string name per line, no space or tab
        self.relation_file = kwargs['relation_file']
        self.n_relation = count_line(self.relation_file)
        # vocab word per line, no space or tab
        self.vocab_file = kwargs['vocab_file']
        self.n_vocab = count_line(self.vocab_file)

        # entity string name \t number of words \t space-separated content
        self.content_file = kwargs['content_file']
        # entity string name \t number of words \t space-separated title content
        self.entity_title_file = kwargs['entity_title_file']
        # relation string name \t number of words \t space-separated title content
        self.relation_title_file = kwargs['relation_title_file']
        # entity string name per line, no space or tab
        self.avoid_entity_file = kwargs['avoid_entity_file']

        # word [ ] " ".join(embedding values)
        self.word_embed_file = kwargs['word_embed_file']

        # head string name \t relation string name \t space-separated tail string names
        self.training_target_tail_file = kwargs['training_target_tail_file']
        # head string name \t relation string name per line
        self.training_target_tail_key_file = kwargs['training_target_tail_key_file']
        # The rest are similar
        self.training_target_head_file = kwargs['training_target_head_file']
        self.training_target_head_key_file = kwargs['training_target_head_key_file']

        self.evaluation_open_target_tail_file = kwargs['evaluation_open_target_tail_file']
        self.evaluation_closed_target_tail_file = kwargs['evaluation_closed_target_tail_file']
        self.evaluation_target_tail_key_file = kwargs['evaluation_target_tail_key_file']
        self.evaluation_open_target_head_file = kwargs['evaluation_open_target_head_file']
        self.evaluation_closed_target_head_file = kwargs['evaluation_closed_target_head_file']
        self.evaluation_target_head_key_file = kwargs['evaluation_target_head_key_file']

        self.train_file = kwargs['train_file']

        self.NON_TRAINABLE = 'non_trainable'

        self.word_oov = kwargs['word_oov']
        self.word_embedding_size = kwargs['word_embedding_size']
        self.max_content_length = kwargs['max_content_length']

        # This is used for execute extra debugging functions
        if 'debug' in kwargs and kwargs['debug']:
            self.debug = True
        else:
            self.debug = False

        self.non_trainable_scope = None
        with tf.variable_scope('non_trainable') as scp:
            self.non_trainable_scope = scp

        self.embedding_scope = None
        with tf.variable_scope('embeddings') as scp:
            self.embedding_scope = scp

        self.training_input_scope = None
        with tf.variable_scope('training_input') as scp:
            self.training_input_scope = scp

        self.head_scope = None
        with tf.variable_scope('transform_head') as scp:
            self.head_scope = scp

        self.tail_scope = None
        with tf.variable_scope("transform_tail") as scp:
            self.tail_scope = scp

        self.rel_scope = None
        with tf.variable_scope("transform_tail") as scp:
            self.rel_scope = scp

        self.head_rel_scope = None
        with tf.variable_scope("combine_head_rel") as scp:
            self.head_rel_scope = scp

        self.pred_scope = None
        with tf.variable_scope('prediction') as scp:
            self.pred_scope = scp

        self.eval_scope = None
        with tf.variable_scope('eval') as scp:
            self.eval_scope = scp

    def _sanity_check(self, entity_dict: dict, session: tf.Session):
        """ Run a series of sanity check functions to validate the model.

        :param entity_dict: A dictionary of {entity_str_name : entity_numerical_id}
        :param session: A TF session for TF op execution
        :return:
        """

        # Check if the entity_dict matches the TF internal hashmap dict
        ph_ent = tf.placeholder(tf.string, [None], name='sanity_check_str_ent')
        entities, entity_ids = zip(*entity_dict.items())

        lookup_res = session.run(self.entity_table.lookup(ph_ent), feed_dict={ph_ent: entities})

        for i, j in zip(lookup_res, entity_ids):
            assert i == j
        tf.logging.info("Sanity check on given entity->id dict passed.")

    def _init_nontrainable_variables(self, session=None):
        """ Initialize all variables of the TF model.

        This should be called after the model is created, and no
        previous checkpoints are found. This will overrides all
        variables including the lookup tables of words and entities.

        Make sure you did not change the input files of a dataset
        otherwise the numerical ids may not match the ones used in
        training.

        :param session: A TF session for TF op execution.
        :return:
        """

        # Load training triples
        _training_triples = load_triples(self.train_file)
        self.training_triples.load(_training_triples, session)
        del _training_triples

        # Load entity list
        # entity_str_name : numerical id (0-indexed)
        entity_dict = load_list(self.entity_file)
        tf.logging.debug("entity_dict loaded.")
        relation_dict = load_list(self.relation_file)
        tf.logging.debug("relation_dict loaded.")

        # Load mask entity list, these are entities used in open world predictions
        # so we need to make sure we do not use them during training
        # Load entities we want to avoid during
        # training entity_str_name : numerical id (0-indexed)
        mask_entity_keys = load_list(self.avoid_entity_file).keys()
        mask_entity_dict = dict((k, entity_dict[k]) for k in mask_entity_keys)
        self.avoid_entities.load(list(mask_entity_dict.values()), session=session)
        tf.logging.info("avoid_entities size %d" % len(mask_entity_dict))

        # This is the entity set we will use during training
        _closed_entities = intbitset(list(entity_dict.values())).difference(
            intbitset(list(mask_entity_dict.values()))).tolist()
        tf.logging.info("closed_entities size %d" % len(_closed_entities))
        self.closed_entities.load(_closed_entities, session=session)

        # Load entity description
        _entity_desc, _entity_desc_len = load_content(self.content_file,
                                                      entity_dict,
                                                      max_content_len=self.max_content_length)
        tf.logging.debug("content file loaded.")
        self.entity_content.load(_entity_desc, session)
        self.entity_content_len.load(_entity_desc_len, session)
        # Release memory before the function ends
        del _entity_desc, _entity_desc_len
        tf.logging.debug("content hashmap loaded.")

        # Load entity title
        _entity_title, _entity_title_len = load_content(self.entity_title_file,
                                                        entity_dict,
                                                        max_content_len=self.max_content_length)
        tf.logging.debug("title file loaded.")
        self.entity_title.load(_entity_title, session)
        self.entity_title_len.load(_entity_title_len, session)
        tf.logging.debug("title hashmap loaded.")

        # Load relationship title
        _relation_title, _relation_title_len = load_content(self.relation_title_file,
                                                            relation_dict,
                                                            max_content_len=self.max_content_length)
        tf.logging.debug("relation file loaded.")
        self.relation_title.load(_relation_title, session)
        self.relation_title_len.load(_relation_title_len, session)
        tf.logging.debug("relation hashmap loaded.")

        # Note, this file has to have the same order as the key file otherwise this will not work
        # THIS PROGRAM DO NOT DO ANY VALIDATION ON THE INPUT SO MAKE SURE THIS IS CORRECT
        # THE TAIL/HEAD TARGETS FILE SHOULD BE GENERATED BY tools/generate_training_target_files.py

        # given a head and rel, what are the true targets in training file
        _training_tail_targets = load_target_file(self.training_target_tail_file)
        self.training_target_tails.load(_training_tail_targets, session)
        tf.logging.debug("%d training_tail_targets loaded." % len(_training_tail_targets))

        # given a tail and rel, what are the true targets in training file
        _training_head_targets = load_target_file(self.training_target_head_file)
        self.training_target_heads.load(_training_head_targets, session)
        tf.logging.debug("%d training_head_targets loaded." % len(_training_head_targets))

        # initialize the pre-trained word embedding
        self.vocab_dict = load_vocab_file(self.vocab_file)
        self.word_embedding.load(load_vocab_embedding(self.word_embed_file, self.vocab_dict, self.word_oov),
                                 session=session)

        if self.debug:
            self._sanity_check(entity_dict, session=session)
            self._sanity_check(mask_entity_dict, session=session)

    def _create_nontrainable_variables(self):
        """ Non trainable variables/constants.

        The variables created in this function will not be stored
        in the checkpoints. If the input data changes
        (except the training/evaluation data), the hashmaps may
        also change and makes the model predicts the wrong result.

        You can manually save all the non-trainable variables by
        saving the variables in the tf.get_collection(self.NON_TRAINABLE)
        collection.

        Use variables to avoid the 4GB graphDef limitation.

        :return:
        """

        with tf.device('/cpu:0'):
            with tf.variable_scope(self.non_trainable_scope):
                # str names of triples
                _n_training_triples = count_line(self.train_file)
                self.training_triples = tf.get_variable("training_triples",
                                                        initializer=[["", "", ""]] * _n_training_triples,
                                                        dtype=tf.string,
                                                        trainable=False,
                                                        collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] training_triples shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.training_triples.get_shape()
                ))
                # entity string name to id
                # Initialized
                self.entity_table = get_lookup_table(self.entity_file,
                                                     oov_buckets=0,
                                                     size=self.n_entity,
                                                     name='entity_lookup_table')

                # entity mask
                # Need to be initialized
                self.avoid_entities = tf.get_variable("avoid_entities",
                                                      shape=[count_line(self.avoid_entity_file)],
                                                      initializer=tf.zeros_initializer(),
                                                      dtype=tf.int32,
                                                      trainable=False,
                                                      collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] avoid_entities shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.avoid_entities.get_shape()
                ))

                self.closed_entities = tf.get_variable("closed_entities",
                                                       shape=[self.n_entity - int(self.avoid_entities.get_shape()[0])],
                                                       initializer=tf.zeros_initializer(),
                                                       dtype=tf.int32,
                                                       trainable=False,
                                                       collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] closed_entities shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.closed_entities.get_shape()
                ))

                # relationship string name to id
                # Initialized
                self.relation_table = get_lookup_table(self.relation_file,
                                                       oov_buckets=0,
                                                       size=self.n_relation,
                                                       name='relation_lookup_table')

                _str_initializer = [""] * self.n_entity
                # content matrix of each entity
                self.entity_content = tf.get_variable("entity_content_matrix",
                                                      dtype=tf.string,
                                                      initializer=_str_initializer,
                                                      trainable=False,
                                                      collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] entity_content shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.entity_content.get_shape()
                ))
                # content length of each entity
                self.entity_content_len = tf.get_variable("entity_content_len",
                                                          [self.n_entity],
                                                          dtype=tf.int32,
                                                          trainable=False,
                                                          collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] entity_content_len shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.entity_content_len.get_shape()
                ))
                # entity title
                self.entity_title = tf.get_variable("entity_title_matrix",
                                                    dtype=tf.string,
                                                    initializer=_str_initializer,
                                                    trainable=False,
                                                    collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] entity_title shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.entity_title.get_shape()
                ))
                self.entity_title_len = tf.get_variable("entity_title_len",
                                                        [self.n_entity],
                                                        dtype=tf.int32,
                                                        trainable=False,
                                                        collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] entity_title_len shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.entity_title_len.get_shape()
                ))
                # relation title
                self.relation_title = tf.get_variable("relation_title_matrix",
                                                      dtype=tf.string,
                                                      initializer=[""] * self.n_relation,
                                                      trainable=False,
                                                      collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] relation_title shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.relation_title.get_shape()
                ))
                self.relation_title_len = tf.get_variable("relation_title_len",
                                                          [self.n_relation],
                                                          dtype=tf.int32,
                                                          trainable=False,
                                                          collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] relation_title_len shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.relation_title_len.get_shape()
                ))
                self.vocab_table = get_lookup_table(self.vocab_file,
                                                    oov_buckets=self.word_oov,
                                                    size=self.n_vocab,
                                                    name='vocab_lookup_table')

                # target tails, use this to get true targets
                _n_training_target_tails = count_line(self.training_target_tail_file)
                # Need to be initialized
                self.training_target_tails = tf.get_variable("training_target_tails",
                                                             dtype=tf.string,
                                                             initializer=[""] * _n_training_target_tails,
                                                             trainable=False,
                                                             collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] training_target_tails shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.training_target_tails.get_shape()
                ))
                self.training_target_tails_table = get_lookup_table(self.training_target_tail_key_file,
                                                                    oov_buckets=0,
                                                                    name='training_target_tails_lookup_table')

                # target heads, use this to get true targets
                _n_training_target_heads = count_line(self.training_target_head_file)
                # Need to be initialized
                self.training_target_heads = tf.get_variable("training_target_heads",
                                                             dtype=tf.string,
                                                             initializer=[""] * _n_training_target_heads,
                                                             trainable=False,
                                                             collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] training_target_heads shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.training_target_heads.get_shape()
                ))

                self.training_target_heads_table = get_lookup_table(self.training_target_head_key_file,
                                                                    oov_buckets=0,
                                                                    name='training_target_heads_lookup_table')

                self.global_step = tf.Variable(0, trainable=False,
                                               collections=[self.NON_TRAINABLE],
                                               name='global_step')

                self.mean_rank = tf.Variable(0, trainable=False,
                                             collections=[self.NON_TRAINABLE],
                                             name='mean_rank')
                tf.summary.scalar("mean_rank", self.mean_rank, collections=[self.EVAL_SUMMARY])

                self.rank_list = tf.placeholder(tf.int32, shape=None, name='rank_list')
                tf.summary.histogram("rank", self.rank_list, collections=[self.EVAL_SUMMARY])

                # Each unique head rel pair is a query
                self.mrr = tf.Variable(0, trainable=False,
                                       collections=[self.NON_TRAINABLE],
                                       name='mrr')
                tf.summary.scalar("mrr", self.mrr, collections=[self.EVAL_SUMMARY])
                # Each head rel pair is a query
                self.triple_mrr = tf.Variable(0, trainable=False,
                                              collections=[self.NON_TRAINABLE],
                                              name='triple_mrr')
                tf.summary.scalar("triple_mrr", self.triple_mrr, collections=[self.EVAL_SUMMARY])

    def _create_embeddings(self, device='/cpu:0'):
        """ Create all embedding matrices in this function.

        :param device: The storage device of all the embeddings. If
                        you are using multi-gpus, it is ideal to store
                        the embeddings on CPU to avoid costly GPU-to-GPU
                        memory copying. The embeddings should be stored under
                        variable scope self.embedding_scope
        :return:
        """
        with tf.device(device):
            with tf.variable_scope(self.embedding_scope):
                self.word_embedding = tf.get_variable('word_embedding',
                                                      [self.n_vocab + self.word_oov, self.word_embedding_size],
                                                      dtype=tf.float32,
                                                      initializer=layers.xavier_initializer())

    def _create_training_input_pipeline(self, num_epoch=10, batch_size=200, sampled_true=1, sampled_false=10):
        """ Create an input pipeline that provides pre-processed, sampled training data

            The pipeline will be stored on CPU because it can not be optimized by GPU.
        :return:
            corrupt_head: [None] TF boolean
            ent: [None] scalar
            rel: [None] scalar
            true_targets: [None, sampled_true]
            false_targets: [None, sampled_false]
        """

        with tf.device('/cpu:0'):
            # TODO: check if the variable scope is useless because there is no new, global variables here
            with tf.variable_scope(self.training_input_scope):
                input_triples = tf.train.limit_epochs(self.training_triples,
                                                      num_epochs=num_epoch,
                                                      name='limited_training_triples')

                # Extract a single h,r,t triple from the input_triples, this is not corrupted
                single_triple = tf.train.shuffle_batch([input_triples],
                                                       batch_size=1,
                                                       capacity=batch_size * 40,
                                                       min_after_dequeue=batch_size * 35,
                                                       enqueue_many=True,
                                                       num_threads=1,
                                                       allow_smaller_final_batch=False,
                                                       name='single_training_shuffle_batch')

                # For a single triple, corrupt it and generate the corrupted training data
                # Input <h, r, t>
                # A .5 probability of corrupting on either h or t
                # Based on the corruption target,
                #   select num_true positive targets **with replacement** (this is just an approximation
                #       and engineering hack, the correct way should be without replacement.)
                #   and num_sampled negative targets without replacement
                #       and avoid all positive targets and unseen masks (self.avoid_entities)

                # After here the entities and relationships are numerical ids
                corrupt_head, ent, rel, true_targets, false_targets = corrupt_single_entity_w_multiple_targets(
                    single_triple,
                    self.training_target_tails_table,
                    self.training_target_heads_table,
                    self.training_target_tails,
                    self.training_target_heads,
                    self.avoid_entities,
                    self.entity_table,
                    self.relation_table,
                    self.n_entity - 1,
                    sampled_true,
                    sampled_false)

                tf.logging.debug("training pipeline shapes corrupt_head %s, "
                                 "ent %s, rel %s, true_targets %s, false_targets %s" %
                                 tuple([x.get_shape() for x in [corrupt_head, ent, rel,
                                                                true_targets, false_targets]]))

                # This is the actual batch queue that returns `batch_size` number of training samples
                q = tf.train.batch([corrupt_head, ent, rel, true_targets, false_targets],
                                   batch_size=batch_size,
                                   num_threads=1,
                                   capacity=batch_size * 40,
                                   enqueue_many=False,
                                   allow_smaller_final_batch=False,
                                   name='corrupted_training_queue')

                return q

    @staticmethod
    def _entity_word_averaging(content_embedding, content_len,
                               padding_word_embedding, orig_shape, device, name=None):
        """ Calculate the averaging embedding of given entities.

        :param content_embedding: 3 dimension tensor with shape [?, embedding_size]
        :param content_len: 1 dimension vector with the same length as the 0-dim
                            of content_embedding
        :param title_embedding: 3 dimension tensor [?, ?, embedding_size]
        :param title_len: 1 dimension vector with the same length as the 0-dim
                            of title_embedding
        :param padding_word_embedding: The [embedding_size] embedding for padding
        :param orig_shape: The original shape of the input entity's first
                            two dimension. This is used to reconstruct
                            [?,?,embedding_size] tensor
        :param device: device to run these ops.
        :param name: Optional name of this op.
        :return:
            embeddings for each entity with the shape of orig_shape + [embedding_size].
        """
        with tf.name_scope(name, 'entity_word_averaging', [content_embedding, content_len,
                                                           padding_word_embedding, orig_shape]):
            with tf.device(device):
                avg_content_embedding = avg_content(content_embedding, content_len,
                                                    padding_word_embedding,
                                                    name='avg_content_embedding')

                avg_embedding = avg_content_embedding  # + avg_title_embedding
                orig_embedding_shape = tf.concat([orig_shape, tf.shape(avg_embedding)[1:]], axis=0,
                                                 name='orig_head_embedding_shape')

                transformed_ents = tf.reshape(avg_embedding, orig_embedding_shape,
                                              name='transformed_head_embedding')

                tf.logging.debug("[%s] transformed_ents shape %s" % (sys._getframe().f_code.co_name,
                                                                     transformed_ents.get_shape()))

                return transformed_ents

    def _transform_head_entity(self, heads, reuse=True, device='/cpu:0', name=None):
        """ Generate head entity representation using given entity ids.

        :param heads:
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        # TODO: add a new parameter to control if we want to corrupt the input
        with tf.name_scope(name, 'transform_head_entity',
                           [heads, self.word_embedding,
                            self.vocab_table,
                            self.entity_content, self.entity_content_len,
                            self.entity_title, self.entity_title_len]):
            tf.logging.debug("[%s] heads shape %s" % (sys._getframe().f_code.co_name,
                                                      heads.get_shape()))

            with tf.variable_scope(self.head_scope, reuse=reuse):
                flatten_heads = tf.reshape(heads, [-1], name='flatten_heads')
                orig_head_shape = tf.shape(heads, name='orig_head_shape')
                head_content_embedding, head_content_len = entity_content_embedding_lookup(entities=flatten_heads,
                                                                                           content=self.entity_content,
                                                                                           content_len=self.entity_content_len,
                                                                                           vocab_table=self.vocab_table,
                                                                                           word_embedding=self.word_embedding,
                                                                                           str_pad=self.PAD,
                                                                                           name='head_content_embedding_lookup')

                head_title_embedding, head_title_len = entity_content_embedding_lookup(entities=flatten_heads,
                                                                                       content=self.entity_title,
                                                                                       content_len=self.entity_title_len,
                                                                                       vocab_table=self.vocab_table,
                                                                                       word_embedding=self.word_embedding,
                                                                                       str_pad=self.PAD,
                                                                                       name='head_title_embedding_lookup')

                pad_word_embedding = self.word_embedding[tf.cast(self.vocab_table.lookup(self.PAD_const), tf.int32), :]
                transformed_heads = self._entity_word_averaging(content_embedding=head_content_embedding,
                                                                content_len=head_content_len,
                                                                # title_embedding=head_title_embedding,
                                                                # title_len=head_title_len,
                                                                padding_word_embedding=pad_word_embedding,
                                                                orig_shape=orig_head_shape,
                                                                device=device)
                return transformed_heads

    def _transform_tail_entity(self, tails, reuse=True, device='/cpu:0', name=None):
        """

        :param tails: Any shape
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        # TODO: add a new parameter to control if we want to corrupt the input
        with tf.name_scope(name, 'transform_tail_entity',
                           [tails, self.word_embedding,
                            self.vocab_table,
                            self.entity_content, self.entity_content_len,
                            self.entity_title, self.entity_title_len]):
            tf.logging.debug("[%s] heads shape %s" % (sys._getframe().f_code.co_name,
                                                      tails.get_shape()))

            with tf.variable_scope(self.tail_scope, reuse=reuse):
                flatten_tails = tf.reshape(tails, [-1], name='flatten_tails')
                orig_tail_shape = tf.shape(tails, name='orig_tail_shape')
                tail_content_embedding, tail_content_len = entity_content_embedding_lookup(entities=flatten_tails,
                                                                                           content=self.entity_content,
                                                                                           content_len=self.entity_content_len,
                                                                                           vocab_table=self.vocab_table,
                                                                                           word_embedding=self.word_embedding,
                                                                                           str_pad=self.PAD,
                                                                                           name='tail_content_embedding_lookup')

                tail_title_embedding, tail_title_len = entity_content_embedding_lookup(entities=flatten_tails,
                                                                                       content=self.entity_title,
                                                                                       content_len=self.entity_title_len,
                                                                                       vocab_table=self.vocab_table,
                                                                                       word_embedding=self.word_embedding,
                                                                                       str_pad=self.PAD,
                                                                                       name='tail_title_embedding_lookup')
                pad_word_embedding = self.word_embedding[tf.cast(self.vocab_table.lookup(self.PAD_const), tf.int32), :]
                transformed_tails = self._entity_word_averaging(content_embedding=tail_content_embedding,
                                                                content_len=tail_content_len,
                                                                # title_embedding=tail_title_embedding,
                                                                # title_len=tail_title_len,
                                                                padding_word_embedding=pad_word_embedding,
                                                                orig_shape=orig_tail_shape,
                                                                device=device)

                return transformed_tails

    def _transform_relation(self, rels, reuse=True, device='/cpu:0', name=None):
        """

        :param rels: Any shape
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        with tf.name_scope(name, 'transform_relation',
                           [rels, self.word_embedding, self.vocab_table,
                            self.relation_title, self.relation_title_len]):
            tf.logging.debug("[%s] rels shape %s" % (sys._getframe().f_code.co_name,
                                                     rels.get_shape()))

            # Here we assume that input relation is always [?, 1] or [?]
            rels = tf.reshape(rels, [-1], name='flatten_rels')
            orig_rels_shape = tf.shape(rels, name='orig_rels_shape')

            rel_embedding, rel_title_len = entity_content_embedding_lookup(entities=rels,
                                                                           content=self.relation_title,
                                                                           content_len=self.relation_title_len,
                                                                           vocab_table=self.vocab_table,
                                                                           word_embedding=self.word_embedding,
                                                                           str_pad=self.PAD,
                                                                           name='rel_embedding_lookup')

            with tf.device(device):
                avg_rel_embedding = avg_content(rel_embedding, rel_title_len,
                                                self.word_embedding[0, :],
                                                name='avg_rel_embedding')
                orig_rel_embedding_shape = tf.concat([orig_rels_shape, tf.shape(avg_rel_embedding)[1:]], axis=0,
                                                     name='orig_rel_embedding_shape')
                transformed_rels = tf.reshape(avg_rel_embedding, orig_rel_embedding_shape,
                                              name='transformed_tail_embedding')
                tf.logging.debug("[%s] transformed_rels shape %s" % (sys._getframe().f_code.co_name,
                                                                     transformed_rels.get_shape()))
                return transformed_rels

    def _combine_head_relation(self, transformed_heads, transformed_rels, reuse=True, device='/cpu:0', name=None):
        """

        :param transformed_heads: [?, ?, word_dim]
        :param transformed_rels:  [?, word_dim]
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        with tf.name_scope(name, 'combine_head_rel',
                           [transformed_heads, transformed_rels]):
            with tf.variable_scope(self.head_rel_scope, reuse=reuse):
                # we assume that the first dimension of heads and rels are the same
                with tf.device(device):
                    # return transformed_heads + tf.expand_dims(transformed_rels, axis=1)
                    return transformed_heads + tf.expand_dims(transformed_rels, axis=1)

    def _predict(self, combined_head_rel, tails, reuse=True, device='/cpu:0', name=None):
        """

        :param combined_head_rel: [?, ?, word_dim]
        :param tails: [?, ?, word_dim]
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        with tf.name_scope(name, 'predict',
                           [combined_head_rel, tails]):
            with tf.variable_scope(self.pred_scope, reuse=reuse):
                with tf.device(device):
                    combined_head_rel, tails = [tf.check_numerics(normalized_embedding(x), '__predict') for x in
                                                [combined_head_rel, tails]]
                    return tf.reduce_sum(combined_head_rel * tails, axis=-1)

    def translate_triple(self, heads, tails, rels, device, reuse=True):
        with tf.name_scope('translate_triple'):
            tf.logging.debug("[%s] heads: %s tails %s rels %s device %s" % (sys._getframe().f_code.co_name,
                                                                            heads.get_shape(),
                                                                            tails.get_shape(),
                                                                            rels.get_shape(),
                                                                            device))
            transformed_heads = self._transform_head_entity(heads, reuse=reuse, device=device)
            transformed_tails = self._transform_tail_entity(tails, reuse=reuse, device=device)
            transformed_rels = self._transform_relation(rels, reuse=reuse, device=device)

            combined_head_rel = self._combine_head_relation(transformed_heads=transformed_heads,
                                                            transformed_rels=transformed_rels,
                                                            reuse=reuse,
                                                            device=device)

            tf.logging.debug("[%s] transformed_heads: %s "
                             "transformed_tails %s "
                             "transformed_rels %s" % (sys._getframe().f_code.co_name,
                                                      transformed_heads.get_shape(),
                                                      transformed_tails.get_shape(),
                                                      transformed_rels.get_shape()))

            return self._predict(combined_head_rel, transformed_tails, reuse=True, device=device)

    def _train_helper(self, corrupt_head, ent, rel, true_targets, false_targets, device):
        with tf.name_scope('train', [corrupt_head, ent, rel, true_targets, false_targets]):
            # if corrupt_head, then ent is tail
            # if not corrupt_head, then ent is head
            targets = tf.concat([true_targets, false_targets], axis=-1, name='concat_targets')

            tf.logging.debug("[%s] targets shape %s" % (sys._getframe().f_code.co_name,
                                                        targets.get_shape()))

            # Convert [?] to [?, 1]
            rel = tf.expand_dims(rel, axis=1)

            # Here we use corrupt_head to separate the input into two sets, head corrupted and tail corrupted
            # and run the model accordingly

            # This will change the internal order of each example in one mini batch
            # TODO: Performance: Concatenate all inputs together and then split them, in that case
            #   the boolean_mask will be applied only once

            corrupt_tail = tf.logical_not(corrupt_head, name='corrupt_tail')

            corrupt_head_pred_score = self.translate_triple(
                *[tf.boolean_mask(x, corrupt_head) for x in [targets, ent, rel]],
                device=device)

            corrupt_tail_pred_score = self.translate_triple(
                *[tf.boolean_mask(x, corrupt_tail) for x in [ent, targets, rel]],
                device=device)

            tf.logging.debug("[%s] corrupt_head_pred_score shape %s, "
                             "corrupt_tail_pred_score %s" % (sys._getframe().f_code.co_name,
                                                             corrupt_head_pred_score.get_shape(),
                                                             corrupt_tail_pred_score.get_shape()))
            # TODO: Monitor: Here we could add summaries on the scores of corrupt tails and corrupt heads individually
            pred_score = tf.concat([corrupt_head_pred_score, corrupt_tail_pred_score], axis=0)

            tf.logging.debug("[%s] pred_score shape %s" % (sys._getframe().f_code.co_name,
                                                           pred_score.get_shape()))

            return pred_score

    def create(self, device='/cpu:0'):
        self._create_nontrainable_variables()
        self._create_embeddings(device)

        with tf.device(device):
            # Create the model with placeholders, this will make sure the variables
            # are created on the given device.
            ph_ents = tf.placeholder(tf.int32, [None, None], name='ph_ents')
            ph_rels = tf.placeholder(tf.int32, [None], name='ph_rels')

            self.translate_triple(heads=ph_ents,
                                  tails=ph_ents,
                                  rels=ph_rels,
                                  device=device,
                                  reuse=False)

    def initialize(self, session):
        self._init_nontrainable_variables(session)

    def train_ops(self, lr=0.01, num_epoch=10, batch_size=200,
                  sampled_true=1, sampled_false=1, devices=list(['/cpu:0'])):

        # If only running on one device then calculate the grads on that device
        if len(devices) == 1:
            grad_dev = devices[0]
        else:
            grad_dev = '/cpu:0'

        with tf.device(grad_dev):
            optimizer = tf.train.AdamOptimizer(lr)
            tower_grads = list()
            losses = list()
            avg_positive_scores = list()
            avg_negative_scores = list()
            score_margin = list()

        tf.logging.info("gradient device %s" % grad_dev)

        for device in devices:
            with tf.device('/cpu:0'):  # input pipeline is always on CPU
                q = self._create_training_input_pipeline(num_epoch=num_epoch,
                                                         batch_size=batch_size,
                                                         sampled_true=sampled_true,
                                                         sampled_false=sampled_false)

            with tf.device(device):
                pred_score = self._train_helper(*q, device=device)
                # tiling the labels so it has the same shape as pred_score
                _labels = [([1.0 / sampled_true] * sampled_true) + ([0.0] * sampled_false)] * batch_size
                labels = tf.constant(_labels, dtype=tf.float32, name='labels')

                tf.logging.debug("[%s] pred_score %s labels %s" % (sys._getframe().f_code.co_name,
                                                                   pred_score.get_shape(),
                                                                   labels.get_shape()))
                pos_pred, neg_pred = tf.split(pred_score, [sampled_true, sampled_false], axis=1)
                avg_positive_scores.append(tf.reduce_mean(pos_pred))
                avg_negative_scores.append(tf.reduce_mean(neg_pred))
                # minimum gap between the smallest positive value and the largest negative value
                score_margin.append(tf.reduce_min(pos_pred, axis=1) - tf.reduce_max(neg_pred, axis=1))

                # NAN returns after several hundreds iterations, try use another loss function
                # Update: nan is not caused by softmax cross entropy
                # loss = tf.reduce_mean(
                #     -tf.reduce_sum(labels * tf.log(tf.clip_by_value(tf.nn.softmax(pred_score), 1e-10, 1.0)), axis=-1))

                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                               logits=pred_score)

                grads = optimizer.compute_gradients(loss)
                for grad in grads:
                    print(grad)
                tower_grads.append(grads)
                losses.append(loss)

            tf.logging.info("Initialize graph on %s" % device)

        with tf.device(grad_dev):
            grads = avg_grads(tower_grads) if len(tower_grads) > 1 else tower_grads[0]

            # Clip weights for predict_weight
            grads = [(grad, var) if 'predict_weight' not in var.name else (tf.clip_by_value(grad, -1., 1.), var) for
                     grad, var in grads]

            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad, collections=[self.TRAIN_SUMMARY_SLOW])

            train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
            loss_op = tf.reduce_mean(tf.stack(losses))

            tf.summary.scalar("loss", loss_op, collections=[self.TRAIN_SUMMARY])
            tf.summary.scalar("avg_pos_score", tf.reduce_mean(tf.stack(avg_positive_scores)),
                              collections=[self.TRAIN_SUMMARY])
            tf.summary.scalar("avg_neg_score", tf.reduce_mean(tf.stack(avg_negative_scores)),
                              collections=[self.TRAIN_SUMMARY])
            tf.summary.histogram("margin", tf.stack(score_margin), collections=[self.TRAIN_SUMMARY_SLOW])
            merge_op = tf.summary.merge_all(key=self.TRAIN_SUMMARY)
            slow_merge_op = tf.summary.merge_all(key=self.TRAIN_SUMMARY_SLOW)

        return train_op, loss_op, [merge_op, slow_merge_op]

    def _eval_targets(self, heads, rels, tails, targets, device, name=None):
        """ For a set of targets, calculate heads, rels -> targets and targets, rels -> tails

        :param heads:
        :param rels:
        :param tails:
        :param targets:
        :param device:
        :param name:
        :return: [batch_size, #targets] for head and tail prediction
        """
        tf.logging.debug("[%s] %s heads %s "
                         "rels %s "
                         "tails %s"
                         "targets %s" % (sys._getframe().f_code.co_name,
                                         name if name is not None else "eval_targets",
                                         heads.get_shape(),
                                         rels.get_shape(),
                                         tails.get_shape(),
                                         targets.get_shape()))

        with tf.name_scope(name, "eval_targets", [heads, rels, tails, targets]):
            # targets = tf.expand_dims(targets, [-1])
            pred_tails = self.translate_triple(heads=heads,
                                               rels=rels,
                                               tails=targets,
                                               device=device)
            pred_heads = self.translate_triple(heads=targets,
                                               rels=rels,
                                               tails=tails,
                                               device=device)

            tf.logging.debug("[%s] %s pred_heads %s "
                             "pred_tails %s" % (sys._getframe().f_code.co_name,
                                                name if name is not None else "eval_targets",
                                                pred_heads.get_shape(),
                                                pred_tails.get_shape()))

            return pred_heads, pred_tails

    @staticmethod
    def _true_target_helper(entity, relation, targets_lookup_table, entity_table, targets, name=None):
        with tf.name_scope(name, 'true_targets_lookup',
                           [entity, relation, targets_lookup_table, entity_table, targets]):
            t = get_target_entities(entity=entity,
                                    relation=relation,
                                    targets_lookup_table=targets_lookup_table,
                                    entity_table=entity_table,
                                    targets=targets)
            t_mask = tf.SparseTensor(t.indices,
                                     tf.cast(tf.clip_by_value(t.values, -1, 0), tf.float32) * 1e10,
                                     t.dense_shape)
            t_dense = tf.sparse_tensor_to_dense(t, default_value=0, name='true_targets_w_padding')

            return t_dense, t_mask

    def _eval_padded_targets(self, heads, rels, tails, masks, device):
        with tf.name_scope("eval_padded_targets", values=[heads, rels, tails, masks]):
            tf.logging.debug("[%s] heads %s "
                             "rels %s "
                             "tails %s"
                             "masks %s" % (sys._getframe().f_code.co_name,
                                           heads.get_shape(),
                                           rels.get_shape(),
                                           tails.get_shape(),
                                           masks.get_shape()))

            pred_score = self.translate_triple(heads=heads,
                                               rels=rels,
                                               tails=tails,
                                               device=device)
            masked_pred_score = tf.sparse_add(pred_score, masks)
            return masked_pred_score

    @staticmethod
    def _calculate_rank(target_scores, pred_scores):
        with tf.name_scope("calculate_rank", values=[target_scores, pred_scores]):
            ranks = tf.reduce_sum(tf.cast(tf.greater(target_scores, pred_scores), tf.int32), axis=1)
            eqs = tf.reduce_sum(tf.cast(tf.equal(target_scores, pred_scores), tf.int32), axis=1)
            return ranks, eqs

    @staticmethod
    def entity_in_set_indicator(ents, all_ents, name=None):
        with tf.name_scope(name, 'entity_in_set_indicator', [ents, all_ents]):
            if ents.dtype == tf.int64:
                ents = tf.cast(ents, tf.int32)
            # The output intersect's indices are in [0, x] format because there is only one row
            intersect = tf.sets.set_intersection(tf.reshape(ents, [1, -1]), tf.reshape(all_ents, [1, -1]))
            sparse_idx = tf.reshape(intersect.indices[:, 1], [-1, 1])
            tf.logging.info("sparse_idx index shape %s" % sparse_idx.get_shape())
            indicator_value = tf.ones_like(intersect.values, dtype=tf.bool)
            tf.logging.info("intersect indicator_value shape %s" % indicator_value.get_shape())

            indicator = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=sparse_idx,
                                                                  values=indicator_value,
                                                                  dense_shape=tf.shape(ents, out_type=tf.int64)[:1]),
                                                  default_value=False,
                                                  name='ent_set_indicator')
            return indicator

    def manual_eval_tail_ops_v2(self, device='/cpu:0'):
        """ Manually evaluate one single partial triple with a given set of targets

        This function will reduce the computation by reusing the targets of the same
        relationships.

        To use this method, first calculate the transformed tails of all the targets
        and put them into a pipeline, then for each head, rel pair we fetch these precomputed
        target representations and do the calculation to get the similarity score.

        After we evaluated one type of relationship, one needs to manually clean up
        the queue so it can be reused by next relationship.

        :param device:
        :return:
        """

        with tf.name_scope("manual_evaluation_v2"):
            with tf.device(device):
                # the input head, rel pair to evaluate
                ph_head_rel = tf.placeholder(tf.string, [1, 2], name='ph_head_rel')
                # tail targets to evaluate, this can be just part of the total targets
                ph_eval_targets = tf.placeholder(tf.string, [1, None], name='ph_eval_targets')
                # indices of true tail targets in the overall target list
                ph_true_target_idx = tf.placeholder(tf.int32, [None], name='ph_true_target_idx')
                # indices of true targets in the evaluation set
                ph_test_target_idx = tf.placeholder(tf.int32, [None], name='ph_test_target_idx')

                ph_target_size = tf.placeholder(tf.int32, (), name='ph_target_size')

                # A temporary queue for precomputed tails
                pre_computed_tail_queue = tf.FIFOQueue(1000000, dtypes=tf.float32,
                                                       shapes=[[self.word_embedding_size]],
                                                       # This may needs to be change later
                                                       name='tail_queue')

                # Convert string targets to numerical ids
                eval_tails = self.entity_table.lookup(ph_eval_targets)
                # computed tails [1, ?, word_dim]
                computed_tails = tf.squeeze(self._transform_tail_entity(eval_tails, reuse=True, device=device), axis=0)

                # put pre-computed tails into target queue
                # Call this to pre-compute tails for a certain relationship
                pre_compute_tails = pre_computed_tail_queue.enqueue_many(computed_tails)

                # get pre-computed tails from target queue
                dequeue_op = pre_computed_tail_queue.dequeue_many(ph_target_size)
                tail_embeds = tf.expand_dims(dequeue_op, axis=0)
                tf.logging.debug("tail_embeds shape %s" % tail_embeds.get_shape())
                # Put tails back into the queue (this will run after tails are dequeued)
                with tf.control_dependencies([dequeue_op]):
                    re_enqueue = pre_computed_tail_queue.enqueue_many(dequeue_op)

                # First, convert string to indices
                str_heads, str_rels = tf.unstack(ph_head_rel, axis=1)
                heads = self.entity_table.lookup(str_heads)
                rels = self.relation_table.lookup(str_rels)

                # Calculate heads and tails
                computed_heads = self._transform_head_entity(heads, reuse=True, device=device)
                computed_rels = self._transform_relation(rels, reuse=True, device=device)
                combined_head_rel = self._combine_head_relation(transformed_heads=computed_heads,
                                                                transformed_rels=computed_rels,
                                                                reuse=True,
                                                                device=device)

                # This is the score of all the targets given a single partial triple
                pred_scores = tf.reshape(self._predict(combined_head_rel,
                                                       tail_embeds,
                                                       reuse=True,
                                                       device=device), [-1, 1])

                tf.logging.debug("eval pred_scores %s" % pred_scores.get_shape())

                ranks, rr = self.eval_helper(pred_scores, ph_test_target_idx, ph_true_target_idx)

                rand_ranks, rand_rr = self.eval_helper(
                    tf.random_uniform(tf.shape(pred_scores), minval=-1, maxval=1, dtype=tf.float32),
                    ph_test_target_idx, ph_true_target_idx)

                top_10_score, top_10 = tf.nn.top_k(tf.reshape(pred_scores, [-1]), k=10)

                return ph_head_rel, ph_eval_targets, ph_target_size, pre_computed_tail_queue.size(), \
                       ph_true_target_idx, ph_test_target_idx, \
                       pre_compute_tails, re_enqueue, dequeue_op, ranks, rr, rand_ranks, rand_rr, pred_scores, top_10, top_10_score

    def manual_eval_head_ops_v2(self, device='/cpu:0'):
        """ Manually evaluate one single partial triple with a given set of targets

        This function will reduce the computation by reusing the targets of the same
        relationships.

        To use this method, first calculate the transformed tails of all the targets
        and put them into a pipeline, then for each head, rel pair we fetch these precomputed
        target representations and do the calculation to get the similarity score.

        After we evaluated one type of relationship, one needs to manually clean up
        the queue so it can be reused by next relationship.

        :param device:
        :return:
        """

        with tf.name_scope("manual_evaluation_v2"):
            with tf.device(device):
                # the input head, rel pair to evaluate
                ph_tail_rel = tf.placeholder(tf.string, [1, 2], name='ph_tail_rel')
                # tail targets to evaluate, this can be just part of the total targets
                ph_eval_targets = tf.placeholder(tf.string, [1, None], name='ph_eval_targets')
                # indices of true tail targets in the overall target list
                ph_true_target_idx = tf.placeholder(tf.int32, [None], name='ph_true_target_idx')
                # indices of true targets in the evaluation set
                ph_test_target_idx = tf.placeholder(tf.int32, [None], name='ph_test_target_idx')

                ph_target_size = tf.placeholder(tf.int32, (), name='ph_target_size')

                # A temporary queue for precomputed tails
                pre_computed_head_queue = tf.FIFOQueue(1000000, dtypes=tf.float32,
                                                       shapes=[[self.word_embedding_size]],
                                                       # This may needs to be change later
                                                       name='tail_queue')

                # First, convert string to indices
                str_tails, str_rels = tf.unstack(ph_tail_rel, axis=1)
                tails = self.entity_table.lookup(str_tails)
                rels = self.relation_table.lookup(str_rels)

                # Convert string targets to numerical ids
                eval_heads = self.entity_table.lookup(ph_eval_targets)
                # computed tails [1, ?, word_dim]
                computed_heads = tf.squeeze(self._transform_head_entity(eval_heads, reuse=True, device=device), axis=0)

                # put pre-computed tails into target queue
                # Call this to pre-compute tails for a certain relationship
                pre_compute_heads = pre_computed_head_queue.enqueue_many(computed_heads)

                # get pre-computed tails from target queue
                dequeue_op = pre_computed_head_queue.dequeue_many(ph_target_size)
                head_embeds = tf.expand_dims(dequeue_op, axis=0)
                tf.logging.debug("tail_embeds shape %s" % head_embeds.get_shape())
                # Put tails back into the queue (this will run after tails are dequeued)
                with tf.control_dependencies([dequeue_op]):
                    re_enqueue = pre_computed_head_queue.enqueue_many(dequeue_op)

                # Calculate heads and tails
                computed_tails = self._transform_head_entity(tails, reuse=True, device=device)
                computed_rels = self._transform_relation(rels, reuse=True, device=device)
                combined_head_rel = self._combine_head_relation(transformed_heads=head_embeds,
                                                                transformed_rels=computed_rels,
                                                                reuse=True,
                                                                device=device)

                # This is the score of all the targets given a single partial triple
                pred_scores = tf.reshape(self._predict(combined_head_rel,
                                                       computed_tails,
                                                       reuse=True,
                                                       device=device), [-1, 1])

                tf.logging.debug("eval pred_scores %s" % pred_scores.get_shape())

                ranks, rr = self.eval_helper(pred_scores, ph_test_target_idx, ph_true_target_idx)

                rand_ranks, rand_rr = self.eval_helper(
                    tf.random_uniform(tf.shape(pred_scores), minval=-1, maxval=1, dtype=tf.float32),
                    ph_test_target_idx, ph_true_target_idx)

                top_10_score, top_10 = tf.nn.top_k(tf.reshape(pred_scores, [-1]), k=10)

                return ph_tail_rel, ph_eval_targets, ph_target_size, pre_computed_head_queue.size(), \
                       ph_true_target_idx, ph_test_target_idx, \
                       pre_compute_heads, re_enqueue, dequeue_op, ranks, rr, rand_ranks, rand_rr, pred_scores, top_10, top_10_score

    @staticmethod
    def eval_helper(scores, test_target_idx, true_target_idx):
        # [?, 1] scores of each true target in evaluation set
        eval_target_scores = tf.nn.embedding_lookup(scores, test_target_idx)

        tf.logging.debug("eval_target_scores %s" % eval_target_scores)

        # [?, 2] idx are [idx, 0]
        true_target_mask_idx = tf.transpose(tf.stack([tf.cast(true_target_idx, tf.int64),
                                                      tf.zeros_like(true_target_idx, dtype=tf.int64)],
                                                     axis=0))

        # [?, 1]
        true_target_mask = tf.SparseTensor(indices=true_target_mask_idx,
                                           values=tf.ones_like(true_target_idx, dtype=tf.float32) * (
                                               -1e10),
                                           dense_shape=tf.shape(scores, out_type=tf.int64))

        tf.logging.debug("true_target_mask %s" % true_target_mask)

        # apply true_target_mask on to pred_scores, [None, 1]
        masked_scores = scores + tf.sparse_tensor_to_dense(true_target_mask)

        # extracted ranks for each evaluation target [1, None] > [None, 1] => [None, None] => [None]
        ranks = tf.reduce_sum(
            tf.cast(tf.greater(tf.transpose(masked_scores), eval_target_scores), tf.int32), axis=-1) + 1
        rr = 1.0 / tf.cast(tf.reduce_min(ranks), tf.float32)

        return ranks, rr

    def manual_eval_ops(self, device='/cpu:0'):
        """ Manually evaluate one single partial triple with a given set of targets

            Given a <head, rel> pair, a set of evaluation targets, and a set of positive targets
             calculate the filtered rank of each positive targets and return them

        :param device:
        :return:
        """

        with tf.name_scope("manual_evaluation"):
            with tf.device('/cpu:0'):
                # head rel pair to evaluate
                ph_head_rel = tf.placeholder(tf.string, [1, 2], name='ph_head_rel')
                # tail targets to evaluate
                ph_eval_targets = tf.placeholder(tf.string, [1, None], name='ph_eval_targets')
                # indices of true tail targets in ph_eval_targets. Mask these when calculating filtered mean rank
                ph_true_target_idx = tf.placeholder(tf.int32, [None], name='ph_true_target_idx')
                # indices of true targets in the evaluation set, we will return the ranks of these targets
                ph_test_target_idx = tf.placeholder(tf.int32, [None], name='ph_test_target_idx')

                # First, convert string to indices
                str_heads, str_rels = tf.unstack(ph_head_rel, axis=1)
                tf.logging.debug("str_heads %s str_rels %s" % (str_heads.get_shape(), str_rels.get_shape()))

                heads = self.entity_table.lookup(str_heads)
                rels = self.relation_table.lookup(str_rels)

                eval_tails = self.entity_table.lookup(ph_eval_targets)

                tf.logging.debug("heads %s rels %s eval_tails %s " % (heads.get_shape(),
                                                                      rels.get_shape(),
                                                                      eval_tails.get_shape()))

                # Get predicted score of the given partial triple and targets
                # [None, 1]
                pred_scores = tf.reshape(self.translate_triple(heads=heads,
                                                               rels=rels,
                                                               tails=eval_tails,
                                                               device=device), [-1, 1])

                pred_scores_queue = tf.FIFOQueue(1000000, dtypes=tf.float32, shapes=[[1]], name='pred_scores_queue')

                enqueue_op = pred_scores_queue.enqueue_many(pred_scores)

                dequeue_op = pred_scores_queue.dequeue_many(pred_scores_queue.size())

                tf.logging.debug("pred_scores %s" % dequeue_op.get_shape())

                def eval_helper(scores):
                    # [?, 1] scores of each true target in evaluation set
                    eval_target_scores = tf.nn.embedding_lookup(scores, ph_test_target_idx)

                    tf.logging.debug("eval_target_scores %s" % eval_target_scores)

                    # [?, 2] idx are [idx, 0]
                    true_target_mask_idx = tf.transpose(tf.stack([tf.cast(ph_true_target_idx, tf.int64),
                                                                  tf.zeros_like(ph_true_target_idx, dtype=tf.int64)],
                                                                 axis=0))

                    # [?, 1]
                    true_target_mask = tf.SparseTensor(indices=true_target_mask_idx,
                                                       values=tf.ones_like(ph_true_target_idx, dtype=tf.float32) * (
                                                           -1e10),
                                                       dense_shape=tf.shape(scores, out_type=tf.int64))

                    tf.logging.debug("true_target_mask %s" % true_target_mask)

                    # apply true_target_mask on to pred_scores, [None, 1]
                    masked_scores = scores + tf.sparse_tensor_to_dense(true_target_mask)

                    # extracted ranks for each evaluation target [1, None] > [None, 1] => [None, None] => [None]
                    ranks = tf.reduce_sum(
                        tf.cast(tf.greater(tf.transpose(masked_scores), eval_target_scores), tf.int32), axis=-1) + 1
                    rr = 1.0 / tf.cast(tf.reduce_min(ranks), tf.float32)

                    return ranks, rr

                ranks, rr = eval_helper(dequeue_op)

                rand_ranks, rand_rr = eval_helper(tf.random_uniform(tf.stack([pred_scores_queue.size(), 1], axis=0),
                                                                    minval=-1, maxval=1, dtype=tf.float32))

                return ph_head_rel, ph_eval_targets, ph_true_target_idx, \
                       ph_test_target_idx, enqueue_op, ranks, rr, rand_ranks, rand_rr, dequeue_op


def main(_):
    import os
    import sys
    tf.logging.set_verbosity(tf.logging.INFO)
    CHECKPOINT_DIR = sys.argv[1]
    dataset_dir = sys.argv[2]

    is_train = not FLAGS.eval

    model = ContentAveragingModel(entity_file=os.path.join(dataset_dir, 'entities.txt'),
                                  relation_file=os.path.join(dataset_dir, 'relations.txt'),
                                  vocab_file=os.path.join(dataset_dir, 'vocab.txt'),
                                  word_embed_file=os.path.join(dataset_dir, 'embed.txt'),
                                  content_file=os.path.join(dataset_dir, 'descriptions.txt'),
                                  entity_title_file=os.path.join(dataset_dir, 'entity_names.txt'),
                                  relation_title_file=os.path.join(dataset_dir, 'relation_names.txt'),
                                  avoid_entity_file=os.path.join(dataset_dir, 'avoid_entities.txt'),

                                  training_target_tail_file=os.path.join(dataset_dir, 'train.tails.values'),
                                  training_target_tail_key_file=os.path.join(dataset_dir, 'train.tails.idx'),
                                  training_target_head_file=os.path.join(dataset_dir, 'train.heads.values'),
                                  training_target_head_key_file=os.path.join(dataset_dir, 'train.heads.idx'),

                                  evaluation_open_target_tail_file=os.path.join(dataset_dir, 'eval.tails.values.open'),
                                  evaluation_closed_target_tail_file=os.path.join(dataset_dir,
                                                                                  'eval.tails.values.closed'),
                                  evaluation_target_tail_key_file=os.path.join(dataset_dir, 'eval.tails.idx'),

                                  evaluation_open_target_head_file=os.path.join(dataset_dir, 'eval.heads.values.open'),
                                  evaluation_closed_target_head_file=os.path.join(dataset_dir,
                                                                                  'eval.heads.values.closed'),
                                  evaluation_target_head_key_file=os.path.join(dataset_dir, 'eval.heads.idx'),

                                  train_file=os.path.join(dataset_dir, 'train.txt'),

                                  word_oov=100,
                                  word_embedding_size=200,
                                  max_content_length=FLAGS.max_content,
                                  debug=True)

    model.create('/cpu:0')
    if is_train:
        train_op, loss_op, merge_ops = model.train_ops(lr=1e-4, num_epoch=200, batch_size=FLAGS.batch,
                                                       sampled_true=1, sampled_false=4,
                                                       devices=['/gpu:0', '/gpu:1', '/gpu:2'])
    else:
        tf.logging.info("Evaluate mode")

    ph_head_rel, ph_eval_targets, ph_target_size, q_size, ph_true_target_idx, \
    ph_test_target_idx, pre_compute_tails, re_enqueue, dequeue_op, ranks, rr, rand_ranks, rand_rr, _, top_10_id, top_10_score = model.manual_eval_tail_ops_v2(
        '/gpu:0')

    ph_tail_rel_, ph_eval_targets_, ph_target_size_, q_size_, ph_true_target_idx_, \
    ph_test_target_idx_, pre_compute_heads_, re_enqueue_, dequeue_op_, ranks_, rr_, rand_ranks_, rand_rr_, _, top_10_id_, top_10_score_ = model.manual_eval_head_ops_v2(
        '/gpu:0')

    # metric_reset_op = tf.variables_initializer([i for i in tf.local_variables() if 'streaming_metrics' in i.name])
    metric_merge_op = tf.summary.merge_all(model.EVAL_SUMMARY)

    EVAL_BATCH = 2000

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    with tf.Session(config=config) as sess:

        # initialize all variables
        sess.run([tf.tables_initializer(),
                  tf.global_variables_initializer(),
                  # Manually initialize non trainable variables because
                  # these are not included in the global_variables set
                  tf.variables_initializer(tf.get_collection(model.NON_TRAINABLE)),
                  tf.local_variables_initializer()])
        # load variable values from disk
        tf.logging.debug("Non trainable variables %s" % [x.name for x in tf.get_collection(model.NON_TRAINABLE)])
        tf.logging.debug("Global variables %s" % [x.name for x in tf.global_variables()])
        tf.logging.debug("Local variables %s" % [x.name for x in tf.local_variables()])

        model.initialize(sess)

        # queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Evaluation targets
        # avoid_targets = intbitset(sess.run(model.avoid_entities).tolist())
        # tf.logging.info("avoid targets %s" % avoid_targets)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables() + [model.global_step])

        try:
            if os.path.exists(os.path.join(CHECKPOINT_DIR, 'checkpoint')):
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(CHECKPOINT_DIR))
                tf.logging.info("Restored model@%d" % sess.run(model.global_step))
        except tf.errors.NotFoundError:
            tf.logging.error("You may have changed your model and there "
                             "are new variables that can not be load from previous snapshot. "
                             "We will keep running but be aware that parts of your model are"
                             " RANDOM MATRICES!")
        tf.logging.info("Start training...")

        def head_eval_helper(is_test=True, target_filter=True, eval_subset=set(), closed=False):
            tf.logging.info("target_filter %s" % target_filter)

            # sess.run(model.is_train.assign(False))
            # print(sess.run(model.is_train))

            eval_file = 'test.txt' if is_test else 'valid.txt'
            load_func = load_manual_evaluation_file_by_rel if not closed else load_closed_manual_evaluation_file_by_rel
            evaluation_data = load_func(os.path.join(dataset_dir, eval_file),
                                        os.path.join(dataset_dir, 'avoid_entities.txt'),
                                        eval_tail=False)
            if target_filter:
                relation_specific_targets = load_relation_specific_targets(
                    os.path.join(dataset_dir, 'train.tails.idx'),
                    os.path.join(dataset_dir, 'relations.txt')
                )
            else:
                relation_specific_targets = None
                eval_targets_set = load_train_entities(os.path.join(dataset_dir, 'entities.txt'),
                                                       os.path.join(dataset_dir, 'avoid_entities.txt'))

            filtered_targets = load_filtered_targets(os.path.join(dataset_dir, 'eval.heads.idx'),
                                                     os.path.join(dataset_dir, 'eval.heads.values.closed'))

            fieldnames = ['relationship', 'hits10', 'mean_rank', 'mrr', 'mrr_per_triple', 'rand_hits10',
                          'rand_mean_rank', 'rand_mrr',
                          'rand_mrr_per_triple', 'miss', 'triples', 'targets']

            csvfile = open(os.path.join(CHECKPOINT_DIR, 'eval.%d.head.csv' % sess.run(model.global_step)), 'w',
                           newline='')
            csv_writer = csv.DictWriter(csvfile, fieldnames)
            csv_writer.writeheader()

            all_ranks = list()
            all_rr = list()
            all_multi_rr = list()

            random_ranks = list()
            random_rr = list()
            random_multi_rr = list()

            missed = 0
            trips = 0
            for c, rel_str in enumerate(evaluation_data.keys()):
                if len(eval_subset) != 0 and rel_str not in eval_subset:
                    tf.logging.debug("skip relation %s" % rel_str)
                    continue
                if not target_filter:
                    tf.logging.debug("eval_targets_set size %d" % len(eval_targets_set))
                else:
                    if rel_str not in relation_specific_targets:
                        tf.logging.warning("Relation %s does not have any valid targets!" % rel_str)
                        continue
                        # First pre-compute the target embeddings
                    eval_targets_set = relation_specific_targets[rel_str]

                eval_targets = list(eval_targets_set)

                if len(eval_targets_set) == 0:
                    tf.logging.warning("eval_targets_set is empty!")
                    continue

                tf.logging.debug("\nRelation %s : %d" % (rel_str, len(eval_targets)))
                start = 0
                while start < len(eval_targets):
                    end = min(start + EVAL_BATCH, len(eval_targets))
                    sess.run(pre_compute_heads_, feed_dict={ph_tail_rel_: [[rel_str, rel_str]],
                                                            ph_eval_targets_: [eval_targets[start:end]]})

                    start = end
                assert sess.run(q_size_) == len(eval_targets)

                rel_ranks = list()
                rel_rr = list()
                rel_multi_rr = list()
                rel_random_ranks = list()
                rel_random_rr = list()
                rel_random_multi_rr = list()
                rel_miss = 0
                rel_trips = 0

                for tail_str, eval_true_targets_set in evaluation_data[rel_str].items():
                    tail_rel = [[tail_str, rel_str]]
                    tail_rel_str = "\t".join([tail_str, rel_str])
                    true_targets = set(filtered_targets[tail_rel_str]).intersection(eval_targets_set)
                    eval_true_targets = set.intersection(eval_targets_set, eval_true_targets_set)

                    rel_miss += len(eval_true_targets_set) - len(eval_true_targets)
                    missed += len(eval_true_targets_set) - len(eval_true_targets)

                    test_target_idx = sorted([eval_targets.index(x) for x in eval_true_targets])
                    true_target_idx = sorted([eval_targets.index(x) for x in true_targets])

                    assert len(true_target_idx) >= len(test_target_idx)

                    if len(eval_subset) > 0:
                        _ranks, _rr, _rand_ranks, _rand_rr, _, top_10_id_res = sess.run(
                            [ranks_, rr_, rand_ranks_, rand_rr_, re_enqueue_, top_10_id_],
                            feed_dict={ph_tail_rel_: tail_rel,
                                       ph_target_size_: len(eval_targets_set),
                                       ph_true_target_idx_: true_target_idx,
                                       ph_test_target_idx_: test_target_idx})
                        print("%s, %s: %s %s" % (
                            tail_str, rel_str, [eval_targets[x] for x in top_10_id_res], true_targets))
                    else:
                        _ranks, _rr, _rand_ranks, _rand_rr, _ = sess.run(
                            [ranks_, rr_, rand_ranks_, rand_rr_, re_enqueue_],
                            feed_dict={ph_tail_rel_: tail_rel,
                                       ph_target_size_: len(eval_targets_set),
                                       ph_true_target_idx_: true_target_idx,
                                       ph_test_target_idx_: test_target_idx})

                    assert sess.run(q_size_) == len(eval_targets)

                    if len(_ranks):
                        rel_ranks.extend([float(x) for x in _ranks])
                        all_ranks.extend([float(x) for x in _ranks])
                        all_rr.append(_rr)
                        rel_rr.append(_rr)
                        all_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))
                        rel_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))

                        random_ranks.extend([float(x) for x in _rand_ranks])
                        rel_random_ranks.extend([float(x) for x in _rand_ranks])
                        random_rr.append(_rand_rr)
                        rel_random_rr.append(_rand_rr)
                        random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_trips += len(_ranks)
                        trips += len(_ranks)
                    print("%d/%d %d "
                          "MR %.4f (%.4f) HITS %.4f (%.4f)"
                          "MRR(per head,rel) %.4f (%.4f) "
                          "MRR(per tail) %.4f (%.4f) missed %d" % (
                              c + 1, len(evaluation_data), len(all_ranks),
                              np.mean(all_ranks), np.mean(random_ranks),
                              np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                              np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                              np.mean(all_rr), np.mean(random_rr),
                              np.mean(all_multi_rr), np.mean(random_multi_rr),
                              missed), end='\r')
                    # clean up precomputed targets
                sess.run(dequeue_op_, feed_dict={ph_target_size_: len(eval_targets_set)})
                assert sess.run(q_size_) == 0

                csv_writer.writerow({'relationship': rel_str,
                                     'hits10': np.mean([1.0 if x <= 10 else 0. for x in rel_ranks]),
                                     'mean_rank': np.mean(rel_ranks),
                                     'mrr': np.mean(rel_rr),
                                     'mrr_per_triple': np.mean(rel_multi_rr),
                                     'rand_hits10': np.mean([1.0 if x <= 10 else 0. for x in rel_random_ranks]),
                                     'rand_mean_rank': np.mean(rel_random_ranks),
                                     'rand_mrr': np.mean(rel_random_rr),
                                     'rand_mrr_per_triple': np.mean(rel_random_multi_rr),
                                     'miss': rel_miss,
                                     'triples': rel_trips,
                                     'targets': len(eval_targets_set)})

            print("\n%d "
                  "MR %.4f (%.4f) HITS %.4f (%.4f)"
                  "MRR(per head,rel) %.4f (%.4f) "
                  "MRR(per tail) %.4f (%.4f) missed %d" % (
                      len(all_ranks),
                      np.mean(all_ranks), np.mean(random_ranks),
                      np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                      np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                      np.mean(all_rr), np.mean(random_rr),
                      np.mean(all_multi_rr), np.mean(random_multi_rr),
                      missed))

            print("HITS %.4f MRR %.4f" % (sum([1.0 if x <= 10 else 0. for x in all_ranks]) / (len(all_ranks) + missed),
                                          sum(all_rr) / (len(all_ranks) + missed)))

            csv_writer.writerow({'relationship': 'OVERALL',
                                 'hits10': np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                                 'mean_rank': np.mean(all_ranks),
                                 'mrr': np.mean(all_rr),
                                 'mrr_per_triple': np.mean(all_multi_rr),
                                 'rand_hits10': np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                                 'rand_mean_rank': np.mean(random_ranks),
                                 'rand_mrr': np.mean(random_rr),
                                 'rand_mrr_per_triple': np.mean(random_multi_rr),
                                 'miss': missed,
                                 'triples': trips,
                                 'targets': -1})

            csvfile.close()
            # sess.run(model.is_train.assign(True))
            return np.mean(all_ranks), all_ranks, np.mean(all_rr), np.mean(all_multi_rr)

        def tail_eval_helper(is_test=True, target_filter=True, eval_subset=set(), closed=False):
            tf.logging.info("target_filter %s" % target_filter)
            # Set mode to evaluation
            # sess.run(model.is_train.assign(False))
            # print(sess.run(model.is_train))
            # ph_head_rel, ph_eval_targets, ph_true_target_idx, ph_test_target_idx, ranks, rr

            # First load evaluation data
            # {rel : {head : [tails]}}
            eval_file = 'test.txt' if is_test else 'valid.txt'
            load_func = load_manual_evaluation_file_by_rel if not closed else load_closed_manual_evaluation_file_by_rel
            evaluation_data = load_func(os.path.join(dataset_dir, eval_file),
                                        os.path.join(dataset_dir, 'avoid_entities.txt'))
            tf.logging.info("Number of relationships in the evaluation file %d" % len(evaluation_data))

            if target_filter:
                relation_specific_targets = load_relation_specific_targets(
                    os.path.join(dataset_dir, 'train.heads.idx'),
                    os.path.join(dataset_dir, 'relations.txt'))
            else:
                relation_specific_targets = None
                eval_targets_set = load_train_entities(os.path.join(dataset_dir, 'entities.txt'),
                                                       os.path.join(dataset_dir, 'avoid_entities.txt'))

            filtered_targets = load_filtered_targets(os.path.join(dataset_dir, 'eval.tails.idx'),
                                                     os.path.join(dataset_dir, 'eval.tails.values.closed'))

            # res_file = open(os.path.join(CHECKPOINT_DIR, 'top_10_score.csv' % sess.run(model.global_step)), 'w', newline='')
            # res_csv_writer = csv.DictWriter(res_file, ['rel', 'score', 'positive'])
            # res_csv_writer.writeheader()

            fieldnames = ['relationship', 'hits10', 'mean_rank', 'mrr', 'mrr_per_triple', 'rand_hits10',
                          'rand_mean_rank', 'rand_mrr',
                          'rand_mrr_per_triple', 'miss', 'triples', 'targets']
            csvfile = open(os.path.join(CHECKPOINT_DIR, 'eval.%d.csv' % sess.run(model.global_step)), 'w', newline='')
            csv_writer = csv.DictWriter(csvfile, fieldnames)
            csv_writer.writeheader()

            all_ranks = list()
            all_rr = list()
            all_multi_rr = list()

            random_ranks = list()
            random_rr = list()
            random_multi_rr = list()

            # Randomly assign some values to the targets, and then run the evaluation

            # New evaluation method - evaluate by relationship
            missed = 0
            trips = 0
            for c, rel_str in enumerate(evaluation_data.keys()):
                if len(eval_subset) != 0 and rel_str not in eval_subset:
                    tf.logging.debug("skip relation %s" % rel_str)
                    continue
                if not target_filter:
                    tf.logging.debug("eval_targets_set size %d" % len(eval_targets_set))
                else:
                    if rel_str not in relation_specific_targets:
                        tf.logging.warning("Relation %s does not have any valid targets!" % rel_str)
                        continue
                    # First pre-compute the target embeddings
                    eval_targets_set = relation_specific_targets[rel_str]

                eval_targets = list(eval_targets_set)

                if len(eval_targets_set) == 0:
                    tf.logging.warning("eval_targets_set is empty!")
                    continue

                tf.logging.debug("\nRelation %s : %d" % (rel_str, len(eval_targets)))
                start = 0
                while start < len(eval_targets):
                    end = min(start + EVAL_BATCH, len(eval_targets))
                    sess.run(pre_compute_tails, feed_dict={ph_head_rel: [[rel_str, rel_str]],
                                                           ph_eval_targets: [eval_targets[start:end]]})
                    start = end

                assert sess.run(q_size) == len(eval_targets)

                # Performance of a single relationship
                rel_ranks = list()
                rel_rr = list()
                rel_multi_rr = list()
                rel_random_ranks = list()
                rel_random_rr = list()
                rel_random_multi_rr = list()
                rel_miss = 0
                rel_trips = 0
                for head_str, eval_true_targets_set in evaluation_data[rel_str].items():
                    head_rel = [[head_str, rel_str]]
                    head_rel_str = "\t".join([head_str, rel_str])

                    # Find true targets (in train/valid/test) of the given head relation
                    # in the evaluation set and skip all others
                    true_targets = set(filtered_targets[head_rel_str]).intersection(eval_targets_set)

                    # find true evaluation targets in the test set that are in this set
                    eval_true_targets = set.intersection(eval_targets_set, eval_true_targets_set)

                    # how many true targets we missed/filtered out
                    rel_miss += len(eval_true_targets_set) - len(eval_true_targets)
                    missed += len(eval_true_targets_set) - len(eval_true_targets)

                    test_target_idx = sorted([eval_targets.index(x) for x in eval_true_targets])
                    true_target_idx = sorted([eval_targets.index(x) for x in true_targets])

                    assert len(true_target_idx) >= len(test_target_idx)

                    if len(eval_subset) > 0:
                        _ranks, _rr, _rand_ranks, _rand_rr, _, top_10_id_res, top_10_score_res = sess.run(
                            [ranks, rr, rand_ranks, rand_rr, re_enqueue, top_10_id, top_10_score],
                            feed_dict={ph_head_rel: head_rel,
                                       ph_target_size: len(eval_targets_set),
                                       ph_true_target_idx: true_target_idx,
                                       ph_test_target_idx: test_target_idx})
                        print("%s, %s: %s %s" % (
                            head_str, rel_str, [eval_targets[x] for x in top_10_id_res], true_targets))
                    else:
                        _ranks, _rr, _rand_ranks, _rand_rr, _ = sess.run(
                            [ranks, rr, rand_ranks, rand_rr, re_enqueue],
                            feed_dict={ph_head_rel: head_rel,
                                       ph_target_size: len(eval_targets_set),
                                       ph_true_target_idx: true_target_idx,
                                       ph_test_target_idx: test_target_idx})

                        # for top_k, top_k_score in enumerate(top_10_score_res):
                        #     res_csv_writer.writerow({'rel': rel_str,
                        #         'score': top_k_score,
                        #         'positive': top_k + 1 in _ranks})

                    assert sess.run(q_size) == len(eval_targets)

                    if len(_ranks):
                        rel_ranks.extend([float(x) for x in _ranks])
                        all_ranks.extend([float(x) for x in _ranks])
                        all_rr.append(_rr)
                        rel_rr.append(_rr)
                        all_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))
                        rel_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))

                        random_ranks.extend([float(x) for x in _rand_ranks])
                        rel_random_ranks.extend([float(x) for x in _rand_ranks])
                        random_rr.append(_rand_rr)
                        rel_random_rr.append(_rand_rr)
                        random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_trips += len(_ranks)
                        trips += len(_ranks)
                    print("%d/%d %d "
                          "MR %.4f (%.4f) HITS %.4f (%.4f)"
                          "MRR(per head,rel) %.4f (%.4f) "
                          "MRR(per tail) %.4f (%.4f) missed %d" % (
                              c + 1, len(evaluation_data), len(all_ranks),
                              np.mean(all_ranks), np.mean(random_ranks),
                              np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                              np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                              np.mean(all_rr), np.mean(random_rr),
                              np.mean(all_multi_rr), np.mean(random_multi_rr),
                              missed), end='\r')
                    # clean up precomputed targets
                sess.run(dequeue_op, feed_dict={ph_target_size: len(eval_targets_set)})
                assert sess.run(q_size) == 0

                csv_writer.writerow({'relationship': rel_str,
                                     'hits10': np.mean([1.0 if x <= 10 else 0. for x in rel_ranks]),
                                     'mean_rank': np.mean(rel_ranks),
                                     'mrr': np.mean(rel_rr),
                                     'mrr_per_triple': np.mean(rel_multi_rr),
                                     'rand_hits10': np.mean([1.0 if x <= 10 else 0. for x in rel_random_ranks]),
                                     'rand_mean_rank': np.mean(rel_random_ranks),
                                     'rand_mrr': np.mean(rel_random_rr),
                                     'rand_mrr_per_triple': np.mean(rel_random_multi_rr),
                                     'miss': rel_miss,
                                     'triples': rel_trips,
                                     'targets': len(eval_targets_set)})

            print("\n%d "
                  "MR %.4f (%.4f) HITS %.4f (%.4f)"
                  "MRR(per head,rel) %.4f (%.4f) "
                  "MRR(per tail) %.4f (%.4f) missed %d" % (
                      len(all_ranks),
                      np.mean(all_ranks), np.mean(random_ranks),
                      np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                      np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                      np.mean(all_rr), np.mean(random_rr),
                      np.mean(all_multi_rr), np.mean(random_multi_rr),
                      missed))

            print("HITS %.4f MRR %.4f" % (sum([1.0 if x <= 10 else 0. for x in all_ranks]) / (len(all_ranks) + missed),
                                          sum(all_rr) / (len(all_ranks) + missed)))

            csv_writer.writerow({'relationship': 'OVERALL',
                                 'hits10': np.mean([1.0 if x <= 10 else 0. for x in all_ranks]),
                                 'mean_rank': np.mean(all_ranks),
                                 'mrr': np.mean(all_rr),
                                 'mrr_per_triple': np.mean(all_multi_rr),
                                 'rand_hits10': np.mean([1.0 if x <= 10 else 0. for x in random_ranks]),
                                 'rand_mean_rank': np.mean(random_ranks),
                                 'rand_mrr': np.mean(random_rr),
                                 'rand_mrr_per_triple': np.mean(random_multi_rr),
                                 'miss': missed,
                                 'triples': trips,
                                 'targets': -1})
            csvfile.close()
            # res_file.close()
            # sess.run(model.is_train.assign(True))
            return np.mean(all_ranks), all_ranks, np.mean(all_rr), np.mean(all_multi_rr)

        def eval_helper(is_test=True):
            # First load evaluation data
            # {rel : {head : [tails]}}
            eval_file = 'test.txt' if is_test else 'valid.txt'
            evaluation_data = load_manual_evaluation_file_by_rel(os.path.join(dataset_dir, eval_file),
                                                                 os.path.join(dataset_dir, 'avoid_entities.txt'))
            tf.logging.info("Number of relationships in the evaluation file %d" % len(evaluation_data))
            relation_specific_targets = load_relation_specific_targets(
                os.path.join(dataset_dir, 'train.heads.idx'),
                os.path.join(dataset_dir, 'relations.txt'))
            filtered_targets = load_filtered_targets(os.path.join(dataset_dir, 'eval.tails.idx'),
                                                     os.path.join(dataset_dir, 'eval.tails.values.closed'))

            fieldnames = ['relationship', 'mean_rank', 'mrr', 'mrr_per_triple', 'rand_mean_rank', 'rand_mrr',
                          'rand_mrr_per_triple', 'miss', 'triples', 'targets']
            csvfile = open(os.path.join(CHECKPOINT_DIR, 'eval.%d.csv' % sess.run(model.global_step)), 'w', newline='')
            csv_writer = csv.DictWriter(csvfile, fieldnames)
            csv_writer.writeheader()

            all_ranks = list()
            all_rr = list()
            all_multi_rr = list()

            random_ranks = list()
            random_rr = list()
            random_multi_rr = list()

            # Randomly assign some values to the targets, and then run the evaluation

            # New evaluation method - evaluate by relationship
            missed = 0
            trips = 0
            for c, rel_str in enumerate(evaluation_data.keys()):

                if rel_str not in relation_specific_targets:
                    tf.logging.warning("Relation %s does not have any valid targets!" % rel_str)
                    continue
                # First pre-compute the target embeddings
                eval_targets_set = relation_specific_targets[rel_str]
                eval_targets = list(eval_targets_set)

                tf.logging.debug("\nRelation %s : %d" % (rel_str, len(eval_targets)))
                start = 0
                while start < len(eval_targets):
                    end = min(start + EVAL_BATCH, len(eval_targets))
                    sess.run(pre_compute_tails, feed_dict={ph_eval_targets: [eval_targets[start:end]]})
                    start = end

                assert sess.run(q_size) == len(eval_targets)

                # Performance of a single relationship
                rel_ranks = list()
                rel_rr = list()
                rel_multi_rr = list()
                rel_random_ranks = list()
                rel_random_rr = list()
                rel_random_multi_rr = list()
                rel_miss = 0
                rel_trips = 0
                for head_str, eval_true_targets_set in evaluation_data[rel_str].items():
                    head_rel = [[head_str, rel_str]]
                    head_rel_str = "\t".join([head_str, rel_str])

                    # Find true targets (in train/valid/test) of the given head relation
                    # in the evaluation set and skip all others
                    true_targets = set(filtered_targets[head_rel_str]).intersection(eval_targets_set)

                    # find true evaluation targets in the test set that are in this set
                    eval_true_targets = set.intersection(eval_targets_set, eval_true_targets_set)

                    # how many true targets we missed/filtered out
                    rel_miss += len(eval_true_targets_set) - len(eval_true_targets)
                    missed += len(eval_true_targets_set) - len(eval_true_targets)

                    test_target_idx = sorted([eval_targets.index(x) for x in eval_true_targets])
                    true_target_idx = sorted([eval_targets.index(x) for x in true_targets])

                    # assert len(true_target_idx) >= len(test_target_idx)

                    _ranks, _rr, _rand_ranks, _rand_rr, _ = sess.run([ranks, rr, rand_ranks, rand_rr, re_enqueue],
                                                                     feed_dict={ph_head_rel: head_rel,
                                                                                ph_target_size: len(eval_targets_set),
                                                                                ph_true_target_idx: true_target_idx,
                                                                                ph_test_target_idx: test_target_idx})

                    # assert sess.run(q_size) == len(eval_targets)

                    if len(_ranks):
                        rel_ranks.extend([float(x) for x in _ranks])
                        all_ranks.extend([float(x) for x in _ranks])
                        all_rr.append(_rr)
                        rel_rr.append(_rr)
                        all_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))
                        rel_multi_rr.extend([np.max([1.0 / float(x) for x in _ranks])] * len(_ranks))

                        random_ranks.extend([float(x) for x in _rand_ranks])
                        rel_random_ranks.extend([float(x) for x in _rand_ranks])
                        random_rr.append(_rand_rr)
                        rel_random_rr.append(_rand_rr)
                        random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_random_multi_rr.extend([np.max([1.0 / float(x) for x in _rand_ranks])] * len(_rand_ranks))
                        rel_trips += len(_ranks)
                        trips += len(_ranks)
                    print("%d/%d %d "
                          "MR %.4f (%.4f) "
                          "MRR(per head,rel) %.4f (%.4f) "
                          "MRR(per tail) %.4f (%.4f) missed %d" % (
                              c + 1, len(evaluation_data), len(all_ranks),
                              np.mean(all_ranks), np.mean(random_ranks),
                              np.mean(all_rr), np.mean(random_rr),
                              np.mean(all_multi_rr), np.mean(random_multi_rr),
                              missed), end='\r')
                    # clean up precomputed targets
                sess.run(dequeue_op, feed_dict={ph_target_size: len(eval_targets_set)})
                # assert sess.run(q_size) == 0

                csv_writer.writerow({'relationship': rel_str,
                                     'mean_rank': np.mean(rel_ranks),
                                     'mrr': np.mean(rel_rr),
                                     'mrr_per_triple': np.mean(rel_multi_rr),
                                     'rand_mean_rank': np.mean(rel_random_ranks),
                                     'rand_mrr': np.mean(rel_random_rr),
                                     'rand_mrr_per_triple': np.mean(rel_random_multi_rr),
                                     'miss': rel_miss,
                                     'triples': rel_trips,
                                     'targets': len(eval_targets_set)})

            print("\n%d "
                  "MR %.4f (%.4f) "
                  "MRR(per head,rel) %.4f (%.4f) "
                  "MRR(per tail) %.4f (%.4f) missed %d" % (
                      len(all_ranks),
                      np.mean(all_ranks), np.mean(random_ranks),
                      np.mean(all_rr), np.mean(random_rr),
                      np.mean(all_multi_rr), np.mean(random_multi_rr),
                      missed))

            csv_writer.writerow({'relationship': 'OVERALL',
                                 'mean_rank': np.mean(all_ranks),
                                 'mrr': np.mean(all_rr),
                                 'mrr_per_triple': np.mean(all_multi_rr),
                                 'rand_mean_rank': np.mean(random_ranks),
                                 'rand_mrr': np.mean(random_rr),
                                 'rand_mrr_per_triple': np.mean(random_multi_rr),
                                 'miss': missed,
                                 'triples': trips,
                                 'targets': -1})

            csvfile.close()

            return np.mean(all_ranks), all_ranks, np.mean(all_rr), np.mean(all_multi_rr)

        if is_train:
            train_writer = tf.summary.FileWriter(CHECKPOINT_DIR, sess.graph, flush_secs=60)

            try:
                global_step = sess.run(model.global_step)
                while not coord.should_stop():

                    if global_step % 10 == 0:
                        if global_step % 500 == 0:
                            _, loss, global_step, merged, merged_slow = sess.run(
                                [train_op, loss_op, model.global_step, merge_ops[0], merge_ops[1]])
                            train_writer.add_summary(merged, global_step)
                            train_writer.add_summary(merged_slow, global_step)
                        else:
                            _, loss, global_step, merged = sess.run(
                                [train_op, loss_op, model.global_step, merge_ops[0]])
                        train_writer.add_summary(merged, global_step)
                    else:
                        _, loss, global_step = sess.run([train_op, loss_op, model.global_step])

                    print("global_step %d loss %.4f" % (global_step, loss), end='\r')

                    if global_step % 1000 == 0:
                        print("Saving model@%d" % global_step)
                        saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=global_step)
                        print("Saved.")
                        mr, r, mrr, triple_mrr = eval_helper(FLAGS.force_eval)

                        model.mean_rank.load(mr, sess)
                        model.mrr.load(mrr, sess)
                        model.triple_mrr.load(triple_mrr)

                        sess.run(metric_merge_op, feed_dict={model.rank_list: r})

            except tf.errors.OutOfRangeError:
                print("training done")
            finally:
                coord.request_stop()

            coord.join(threads)

            saver.save(sess, os.path.join(CHECKPOINT_DIR, "model.ckpt"), global_step=model.global_step)
            tf.logging.info("Model saved with %d global steps." % sess.run(model.global_step))
        else:
            tail_eval_helper()
            head_eval_helper()


if __name__ == '__main__':
    tf.app.run()
