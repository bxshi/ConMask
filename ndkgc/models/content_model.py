from intbitset import intbitset
import math
import tensorflow.contrib.layers as layers
import tensorflow.contrib.metrics as metrics

from ndkgc.ops import *
from ndkgc.utils import *


class ContentModel(object):
    PAD = '__PAD__'
    PAD_const = tf.constant(PAD, name='pad')

    TRAIN_SUMMARY = 'train_summary'
    TRAIN_SUMMARY_SLOW = 'train_summary_slow'

    EVAL_SUMMARY = 'eval_summary'

    def __init__(self, **kwargs):
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

        self.num_epoch = kwargs['num_epoch']
        self.word_oov = kwargs['word_oov']
        self.word_embedding_size = kwargs['word_embedding_size']

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
        """ Run this if in debug mode
        :param entity_dict:
        :param session:
        :return:
        """
        ph_ent = tf.placeholder(tf.string, [None], name='sanity_check_str_ent')
        entities, entity_ids = zip(*entity_dict.items())

        lookup_res = session.run(self.entity_table.lookup(ph_ent), feed_dict={ph_ent: entities})

        for i, j in zip(lookup_res, entity_ids):
            assert i == j
        tf.logging.info("Sanity check passed.")

    def _init_nontrainable_variables(self, session=None):
        """ Call this if no previous checkpoints are found

        :return:
        """

        # Load training triples
        _training_triples = load_triples(self.train_file)
        self.training_triples.load(_training_triples, session)
        del _training_triples

        # Load entity list
        # entity_str_name : numerical id (0-indexed)
        entity_dict = load_list(self.entity_file)
        relation_dict = load_list(self.relation_file)

        # Load mask entity list, these are entities used in open world predictions
        # so we need to make sure we do not use them during training
        # Load entities we want to avoid during
        # training entity_str_name : numerical id (0-indexed)
        mask_entity_keys = load_list(self.avoid_entity_file).keys()
        mask_entity_dict = dict((k, entity_dict[k]) for k in mask_entity_keys)
        self.avoid_entities.load(list(mask_entity_dict.values()), session=session)
        tf.logging.info("avoid_entities size %d" % len(mask_entity_dict.values()))

        _closed_entities = intbitset(list(entity_dict.values())).difference(
            intbitset(list(mask_entity_dict.values()))).tolist()
        tf.logging.info("closed_entities size %d" % len(_closed_entities))
        self.closed_entities.load(_closed_entities, session=session)

        # Load entity description
        _entity_desc, _entity_desc_len = load_content(self.content_file,
                                                      entity_dict)
        self.entity_content.load(_entity_desc, session)
        self.entity_content_len.load(_entity_desc_len, session)
        # Release memory before the function ends
        del _entity_desc, _entity_desc_len

        # Load entity title
        _entity_title, _entity_title_len = load_content(self.entity_title_file,
                                                        entity_dict)
        self.entity_title.load(_entity_title, session)
        self.entity_title_len.load(_entity_title_len, session)

        # Load relationship title
        _relation_title, _relation_title_len = load_content(self.relation_title_file,
                                                            relation_dict)
        self.relation_title.load(_relation_title, session)
        self.relation_title_len.load(_relation_title_len, session)

        # Note, this file has to have the same order as the key file otherwise this will not work
        _training_tail_targets = load_target_file(self.training_target_tail_file)
        self.training_target_tails.load(_training_tail_targets, session)
        del _training_tail_targets

        _training_head_targets = load_target_file(self.training_target_head_file)
        self.training_target_heads.load(_training_head_targets, session)

        _evaluation_open_tail_targets = load_target_file(self.evaluation_open_target_tail_file)
        self.evaluation_open_target_tails.load(_evaluation_open_tail_targets)
        del _evaluation_open_tail_targets
        _evaluation_closed_tail_targets = load_target_file(self.evaluation_closed_target_tail_file)
        self.evaluation_closed_target_tails.load(_evaluation_closed_tail_targets)
        del _evaluation_closed_tail_targets

        _evaluation_open_head_targets = load_target_file(self.evaluation_open_target_head_file)
        self.evaluation_open_target_heads.load(_evaluation_open_head_targets)
        del _evaluation_open_head_targets
        _evaluation_closed_head_targets = load_target_file(self.evaluation_closed_target_head_file)
        self.evaluation_closed_target_heads.load(_evaluation_closed_head_targets)
        del _evaluation_closed_head_targets

        # initialize the pre-trained word embedding
        self.vocab_dict = load_vocab_file(self.vocab_file)
        self.word_embedding.load(load_vocab_embedding(self.word_embed_file, self.vocab_dict, self.word_oov),
                                 session=session)

        if self.debug:
            self._sanity_check(entity_dict, session=session)
            self._sanity_check(mask_entity_dict, session=session)

    def _create_nontrainable_variables(self):
        """ Non trainable variables/constants.

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

                # The evaluation target files contains all the target information, so this is a superset of
                #  the training target files which only contains the information in the training data
                _n_evaluation_open_target_tails = count_line(self.evaluation_open_target_tail_file)
                _n_evaluation_closed_target_tails = count_line(self.evaluation_closed_target_tail_file)
                # all targets not seen during training
                self.evaluation_open_target_tails = tf.get_variable("evaluation_open_target_tails",
                                                                    dtype=tf.string,
                                                                    initializer=[""] * _n_evaluation_open_target_tails,
                                                                    trainable=False,
                                                                    collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] evaluation_open_target_tails shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.evaluation_open_target_tails.get_shape()
                ))
                # all targets seen during training
                self.evaluation_closed_target_tails = tf.get_variable("evaluation_closed_target_tails",
                                                                      dtype=tf.string,
                                                                      initializer=[
                                                                                      ""] * _n_evaluation_closed_target_tails,
                                                                      trainable=False,
                                                                      collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] evaluation_closed_target_tails shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.evaluation_closed_target_tails.get_shape()
                ))
                self.evaluation_target_tails_table = get_lookup_table(self.evaluation_target_tail_key_file,
                                                                      oov_buckets=0,
                                                                      name='evaluation_target_tails_lookup_table')

                _n_evaluation_open_target_heads = count_line(self.evaluation_open_target_head_file)
                _n_evaluation_closed_target_heads = count_line(self.evaluation_closed_target_head_file)
                # all targets not seen during training
                self.evaluation_open_target_heads = tf.get_variable("evaluation_open_target_heads",
                                                                    dtype=tf.string,
                                                                    initializer=[""] * _n_evaluation_open_target_heads,
                                                                    trainable=False,
                                                                    collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] evaluation_open_target_heads shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.evaluation_open_target_heads.get_shape()
                ))
                # all targets seen during training
                self.evaluation_closed_target_heads = tf.get_variable("evaluation_closed_target_heads",
                                                                      dtype=tf.string,
                                                                      initializer=[
                                                                                      ""] * _n_evaluation_closed_target_heads,
                                                                      trainable=False,
                                                                      collections=[self.NON_TRAINABLE])
                tf.logging.debug("[%s] evaluation_closed_target_heads shape: %s" % (
                    sys._getframe().f_code.co_name,
                    self.evaluation_closed_target_heads.get_shape()
                ))
                # all targets missing during training
                self.evaluation_target_heads_table = get_lookup_table(self.evaluation_target_head_key_file,
                                                                      oov_buckets=0,
                                                                      name='evaluation_target_heads_lookup_table')

                self.global_step = tf.Variable(0, trainable=False,
                                               collections=[self.NON_TRAINABLE],
                                               name='global_step')

    def _create_embeddings(self, device='/cpu:0'):
        with tf.device(device):
            with tf.variable_scope(self.embedding_scope):
                self.word_embedding = tf.get_variable('word_embedding',
                                                      [self.n_vocab + self.word_oov, self.word_embedding_size],
                                                      dtype=tf.float32,
                                                      initializer=layers.xavier_initializer())

    def _create_training_input_pipeline(self, num_epoch=10, batch_size=200, sampled_true=1, sampled_false=10):
        """

        :return:
            corrupt_head: [None] TF boolean
            ent: [None] scalar
            rel: [None] scalar
            true_targets: [None, sampled_true]
            false_targets: [None, sampled_false]
        """

        with tf.device('/cpu:0'):
            # TODO: check if the variable scope is useless because there is no new variables here
            with tf.variable_scope(self.training_input_scope):
                input_triples = tf.train.limit_epochs(self.training_triples,
                                                      num_epochs=num_epoch,
                                                      name='limited_training_triples')

                # Extract a single h,r,t triple from the input_triples
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
                # A .5 probability to corrupt on either h or t
                # Based on the corruption target,
                #   select num_true positive targets **with replacement**
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

                tf.logging.info("training pipeline shapes corrupt_head %s, "
                                "ent %s, rel %s, true_targets %s, false_targets %s" %
                                tuple([x.get_shape() for x in [corrupt_head, ent, rel, true_targets, false_targets]]))

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
                               title_embedding, title_len,
                               padding_word_embedding, orig_shape, device, name=None):
        with tf.name_scope(name, 'entity_word_averaging', [content_embedding, content_len,
                                                           title_embedding, title_len,
                                                           padding_word_embedding, orig_shape]):
            with tf.device(device):
                avg_content_embedding = avg_content(content_embedding, content_len,
                                                    padding_word_embedding,
                                                    name='avg_content_embedding')
                avg_title_embedding = avg_content(title_embedding, title_len,
                                                  padding_word_embedding,
                                                  name='avg_title_embedding')

                avg_embedding = avg_content_embedding + avg_title_embedding
                orig_embedding_shape = tf.concat([orig_shape, tf.shape(avg_embedding)[1:]], axis=0,
                                                 name='orig_head_embedding_shape')

                transformed_ents = tf.reshape(avg_embedding, orig_embedding_shape,
                                              name='transformed_head_embedding')

                tf.logging.debug("[%s] transformed_ents shape %s" % (sys._getframe().f_code.co_name,
                                                                     transformed_ents.get_shape()))

                return transformed_ents

    def __transform_head_entity(self, heads, reuse=True, device='/cpu:0', name=None):
        """

        :param heads: Any shape
        :param reuse:
        :param device:
        :param name:
        :return:
        """
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
                                                                title_embedding=head_title_embedding,
                                                                title_len=head_title_len,
                                                                padding_word_embedding=pad_word_embedding,
                                                                orig_shape=orig_head_shape,
                                                                device=device)
                return transformed_heads

    def __transform_tail_entity(self, tails, reuse=True, device='/cpu:0', name=None):
        """

        :param tails: Any shape
        :param reuse:
        :param device:
        :param name:
        :return:
        """
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
                                                                title_embedding=tail_title_embedding,
                                                                title_len=tail_title_len,
                                                                padding_word_embedding=pad_word_embedding,
                                                                orig_shape=orig_tail_shape,
                                                                device=device)

                return transformed_tails

    def __transform_relation(self, rels, reuse=True, device='/cpu:0', name=None):
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

    def __combine_head_relation(self, transformed_heads, transformed_rels, reuse=True, device='/cpu:0', name=None):
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

    def __predict(self, combined_head_rel, tails, reuse=True, device='/cpu:0', name=None):
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
            tf.logging.info("[%s] heads: %s tails %s rels %s device %s" % (sys._getframe().f_code.co_name,
                                                                           heads.get_shape(),
                                                                           tails.get_shape(),
                                                                           rels.get_shape(),
                                                                           device))
            transformed_heads = self.__transform_head_entity(heads, reuse=reuse, device=device)
            transformed_tails = self.__transform_tail_entity(tails, reuse=reuse, device=device)
            transformed_rels = self.__transform_relation(rels, reuse=reuse, device=device)

            combined_head_rel = self.__combine_head_relation(transformed_heads=transformed_heads,
                                                             transformed_rels=transformed_rels,
                                                             reuse=reuse,
                                                             device=device)

            tf.logging.info("[%s] transformed_heads: %s "
                            "transformed_tails %s "
                            "transformed_rels %s" % (sys._getframe().f_code.co_name,
                                                     transformed_heads.get_shape(),
                                                     transformed_tails.get_shape(),
                                                     transformed_rels.get_shape()))

            return self.__predict(combined_head_rel, transformed_tails, reuse=True, device=device)

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
        tf.logging.info("[%s] %s heads %s "
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

            tf.logging.info("[%s] %s pred_heads %s "
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
            tf.logging.info("[%s] heads %s "
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

    def simple_eval_ops(self, batch_size, device='/cpu:0'):
        """ This is the simplest version of evaluation, the input triples are split into batches
        and evaluated on the entire target space without slicing and batching.

        This is designed for small data sets or evaluating the actual memory usage of the
        evaluation graph.

        :param batch_size:
        :param device:
        :return:
        """
        with tf.name_scope("simple_evaluation"):
            with tf.device(device):
                with tf.device('/cpu:0'):
                    bsize = tf.constant(batch_size, tf.int32, shape=(), name='batch_size')
                    ph_evaluation_triples = tf.placeholder(tf.string, [None, 3], name='ph_triples')
                    triple_queue = tf.FIFOQueue(81920, [tf.string], [[3]])
                    enqueue_triple = triple_queue.enqueue_many(ph_evaluation_triples)

                    # This makes sure the dequeue will not be blocked due to insufficient elements
                    current_bsize = tf.maximum(1, tf.minimum(bsize, triple_queue.size()), name='current_batch_size')
                    # Make sure you checked your number of calls
                    # otherwise this will block the entire system because we do not
                    # have enough data in the pipeline
                    str_triple_batch = triple_queue.dequeue_up_to(current_bsize)

                    str_heads, str_rels, str_tails = tf.unstack(str_triple_batch, axis=1)

                    tf.logging.info("[%s] str_heads %s "
                                    "str_rels %s "
                                    "str_tails %s" % (sys._getframe().f_code.co_name,
                                                      str_heads.get_shape(),
                                                      str_rels.get_shape(),
                                                      str_tails.get_shape()))

                    # Reshape them so they are [b_size, 1]
                    heads, rels, tails = [tf.expand_dims(x, axis=1) for x in triple_id_lookup(str_heads,
                                                                                              str_rels,
                                                                                              str_tails,
                                                                                              self.entity_table,
                                                                                              self.relation_table)]

                    tf.logging.info("[%s] heads %s "
                                    "rels %s "
                                    "tails %s" % (sys._getframe().f_code.co_name,
                                                  heads.get_shape(),
                                                  rels.get_shape(),
                                                  tails.get_shape()))

                # Predict score of given triples
                pred_scores = self.translate_triple(heads, tails, rels, device=device)
                tf.logging.info("[%s] pred_scores shape %s " % (sys._getframe().f_code.co_name,
                                                                pred_scores.get_shape()))

                # Predict score of all targets
                open_targets = tf.expand_dims(self.avoid_entities, axis=0)
                pred_open_heads, pred_open_tails = self._eval_targets(heads=heads,
                                                                      rels=rels,
                                                                      tails=tails,
                                                                      targets=open_targets,
                                                                      device=device,
                                                                      name='pred_open_targets')
                closed_targets = tf.expand_dims(self.closed_entities, axis=0)

                pred_closed_heads, pred_closed_tails = self._eval_targets(heads=heads,
                                                                          rels=rels,
                                                                          tails=tails,
                                                                          targets=closed_targets,
                                                                          device=device,
                                                                          name='pred_cloesd_targets')

                # Predict score of modifiers
                closed_tails, closed_tails_mask = get_true_targets(entity=str_heads, relation=str_rels,
                                                                   targets_lookup_table=self.evaluation_target_tails_table,
                                                                   entity_table=self.entity_table,
                                                                   targets=self.evaluation_closed_target_tails)

                closed_heads, closed_heads_mask = get_true_targets(entity=str_tails, relation=str_rels,
                                                                   targets_lookup_table=self.evaluation_target_heads_table,
                                                                   entity_table=self.entity_table,
                                                                   targets=self.evaluation_closed_target_heads)

                open_tails, open_tails_mask = get_true_targets(entity=str_heads, relation=str_rels,
                                                               targets_lookup_table=self.evaluation_target_tails_table,
                                                               entity_table=self.entity_table,
                                                               targets=self.evaluation_open_target_tails)

                open_heads, open_heads_mask = get_true_targets(entity=str_tails, relation=str_rels,
                                                               targets_lookup_table=self.evaluation_target_heads_table,
                                                               entity_table=self.entity_table,
                                                               targets=self.evaluation_open_target_heads)

                pred_true_open_heads = self._eval_padded_targets(heads=open_heads,
                                                                 rels=rels,
                                                                 tails=tails,
                                                                 masks=open_heads_mask,
                                                                 device=device)
                pred_true_open_tails = self._eval_padded_targets(heads=heads,
                                                                 rels=rels,
                                                                 tails=open_tails,
                                                                 masks=open_tails_mask,
                                                                 device=device)
                pred_true_closed_heads = self._eval_padded_targets(heads=closed_heads,
                                                                   rels=rels,
                                                                   tails=tails,
                                                                   masks=closed_heads_mask,
                                                                   device=device)
                pred_true_closed_tails = self._eval_padded_targets(heads=heads,
                                                                   rels=rels,
                                                                   tails=closed_tails,
                                                                   masks=closed_tails_mask,
                                                                   device=device)
                # Final ranks
                all_open_heads_rank, all_open_heads_eq = self._calculate_rank(pred_open_heads, pred_scores)
                all_open_tails_rank, all_open_tails_eq = self._calculate_rank(pred_open_tails, pred_scores)
                all_closed_heads_rank, all_closed_heads_eq = self._calculate_rank(pred_closed_heads, pred_scores)
                all_closed_tails_rank, all_closed_tails_eq = self._calculate_rank(pred_closed_tails, pred_scores)

                closed_tails_rank, closed_tails_eq = self._calculate_rank(pred_true_closed_tails, pred_scores)
                closed_heads_rank, cloesd_heads_eq = self._calculate_rank(pred_true_closed_heads, pred_scores)
                open_tails_rank, open_tails_eq = self._calculate_rank(pred_true_open_tails, pred_scores)
                open_heads_rank, open_tails_eq = self._calculate_rank(pred_true_open_heads, pred_scores)

                classic_head_rank = all_closed_heads_rank + 1
                classic_filtered_head_rank = classic_head_rank - closed_heads_rank
                classic_tail_rank = all_closed_tails_rank + 1
                classic_filtered_tail_rank = classic_tail_rank - closed_tails_rank

                new_head_rank = all_closed_heads_rank + all_open_heads_rank + 1
                new_filtered_head_rank = new_head_rank - closed_heads_rank - open_heads_rank
                new_tail_rank = all_closed_tails_rank + all_open_tails_rank + 1
                new_filtered_tail_rank = new_tail_rank - closed_tails_rank - open_tails_rank

                tf.logging.info("[%s] classic_head_rank %s "
                                "classic_filtered_head_rank %s "
                                "classic_tail_rank %s "
                                "classic_filtered_tail_rank %s "
                                "new_head_rank %s "
                                "new_filtered_head_rank %s "
                                "new_tail_rank %s "
                                "new_filtered_tail_rank %s " % (sys._getframe().f_code.co_name,
                                                                classic_head_rank.get_shape(),
                                                                classic_filtered_head_rank.get_shape(),
                                                                classic_tail_rank.get_shape(),
                                                                classic_filtered_tail_rank.get_shape(),
                                                                new_head_rank.get_shape(),
                                                                new_filtered_head_rank.get_shape(),
                                                                new_tail_rank.get_shape(),
                                                                new_filtered_tail_rank.get_shape()))

                eval_result = {
                    'closed':
                        {
                            'head': {
                                'rank': classic_head_rank,
                                'frank': classic_filtered_head_rank,
                            },
                            'tail ': {
                                'rank': classic_tail_rank,
                                'frank': classic_filtered_tail_rank,
                            },
                        },
                    'open':
                        {
                            'head': {
                                'rank': new_head_rank,
                                'frank': new_filtered_head_rank,
                            },
                            'tail': {
                                'rank': new_tail_rank,
                                'frank': new_filtered_tail_rank,
                            },
                        },
                }

                return ph_evaluation_triples, enqueue_triple, str_triple_batch, eval_result

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
                tf.logging.info("str_heads %s str_rels %s" % (str_heads.get_shape(), str_rels.get_shape()))

                heads = self.entity_table.lookup(str_heads)
                rels = self.relation_table.lookup(str_rels)

                eval_tails = self.entity_table.lookup(ph_eval_targets)

                tf.logging.info("heads %s rels %s eval_tails %s " % (heads.get_shape(),
                                                                     rels.get_shape(),
                                                                     eval_tails.get_shape()))

                # Get predicted score of the given partial triple and targets
                # [None, 1]
                pred_scores = tf.reshape(self.translate_triple(heads=heads,
                                                               rels=rels,
                                                               tails=eval_tails,
                                                               device=device), [-1, 1])

                tf.logging.info("pred_scores %s" % pred_scores.get_shape())

                # [?, 1] scores of each true target in evaluation set
                eval_target_scores = tf.nn.embedding_lookup(pred_scores, ph_test_target_idx)

                tf.logging.info("eval_target_scores %s" % eval_target_scores)

                # [?, 2] idx are [idx, 0]
                true_target_mask_idx = tf.transpose(tf.stack([tf.cast(ph_true_target_idx, tf.int64),
                                                              tf.zeros_like(ph_true_target_idx, dtype=tf.int64)], axis=0))

                # [?, 1]
                true_target_mask = tf.SparseTensor(indices=true_target_mask_idx,
                                                   values=tf.ones_like(ph_true_target_idx, dtype=tf.float32) * (-1e10),
                                                   dense_shape=tf.shape(pred_scores, out_type=tf.int64))

                tf.logging.info("true_target_mask %s" % true_target_mask)

                # apply true_target_mask on to pred_scores, [None, 1]
                masked_scores = pred_scores + tf.sparse_tensor_to_dense(true_target_mask)

                # extracted ranks for each evaluation target [1, None] > [None, 1] => [None, None] => [None]
                ranks = tf.reduce_sum(tf.cast(tf.greater(tf.transpose(masked_scores), eval_target_scores), tf.int32), axis=-1) + 1
                rr = 1.0 / tf.cast(tf.reduce_min(ranks), tf.float32)

                ph_ent_rel_str = tf.placeholder(tf.string, [1], name='ph_ent_rel_str')
                true_tails = get_true_tails(ph_ent_rel_str,
                                            self.evaluation_target_tails_table,
                                            self.evaluation_closed_target_tails)

                return ph_head_rel, ph_eval_targets, ph_true_target_idx, ph_test_target_idx, ph_ent_rel_str, true_tails, ranks, rr, eval_target_scores, pred_scores, masked_scores

    def auto_eval_ops(self, batch_size, n_splits=50, device='/cpu:0'):
        """ Automatically evaluation on the evaluation matrix
        """
        with tf.name_scope("auto_evaluation"):
            with tf.device('/cpu:0'):
                bsize = tf.constant(batch_size, tf.int32, shape=(), name='batch_size')
                ph_evaluation_triples = tf.placeholder(tf.string, [None, 3], name='ph_triples')
                triple_queue = tf.FIFOQueue(81920, [tf.string], [[3]])
                enqueue_triple = triple_queue.enqueue_many(ph_evaluation_triples)

                # This makes sure the dequeue will not be blocked due to insufficient elements
                current_bsize = tf.maximum(1, tf.minimum(bsize, triple_queue.size()), name='current_batch_size')
                # Make sure you checked your number of calls
                # otherwise this will block the entire system because we do not
                # have enough data in the pipeline
                str_triple_batch = triple_queue.dequeue_up_to(current_bsize)

                # single head, rel tail
                str_heads, str_rels, str_tails = tf.unstack(str_triple_batch, axis=1)
                tf.logging.info("[%s] str_heads %s "
                                "str_rels %s "
                                "str_tails %s" % (sys._getframe().f_code.co_name,
                                                  str_heads.get_shape(),
                                                  str_rels.get_shape(),
                                                  str_tails.get_shape()))

                # single head, tail, rel in numerical format
                # Reshape them so they are [b_size, 1]
                heads, rels, tails = [tf.expand_dims(x, axis=1) for x in triple_id_lookup(str_heads,
                                                                                          str_rels,
                                                                                          str_tails,
                                                                                          self.entity_table,
                                                                                          self.relation_table)]

                # Calculate a boolean vector for head and tails,
                # True if the entity is presented in the training data

                tf.logging.info("[%s] heads %s "
                                "rels %s "
                                "tails %s" % (sys._getframe().f_code.co_name,
                                              heads.get_shape(),
                                              rels.get_shape(),
                                              tails.get_shape()))

                # For the given batch, calculate the score of it
                with tf.device(device):
                    pred_scores = self.translate_triple(heads, tails, rels, device=device)
                tf.logging.info("[%s] pred_scores shape %s " % (sys._getframe().f_code.co_name,
                                                                pred_scores.get_shape()))

                # Now calculate the score of the modifiers so we can get the "filtered" scores

                #

                # Scores of the modifiers
                closed_tails, closed_tails_mask = get_true_targets(entity=str_heads, relation=str_rels,
                                                                   targets_lookup_table=self.evaluation_target_tails_table,
                                                                   entity_table=self.entity_table,
                                                                   targets=self.evaluation_closed_target_tails)

                closed_heads, closed_heads_mask = get_true_targets(entity=str_tails, relation=str_rels,
                                                                   targets_lookup_table=self.evaluation_target_heads_table,
                                                                   entity_table=self.entity_table,
                                                                   targets=self.evaluation_closed_target_heads)

                open_tails, open_tails_mask = get_true_targets(entity=str_heads, relation=str_rels,
                                                               targets_lookup_table=self.evaluation_target_tails_table,
                                                               entity_table=self.entity_table,
                                                               targets=self.evaluation_open_target_tails)

                open_heads, open_heads_mask = get_true_targets(entity=str_tails, relation=str_rels,
                                                               targets_lookup_table=self.evaluation_target_heads_table,
                                                               entity_table=self.entity_table,
                                                               targets=self.evaluation_open_target_heads)

                # TODO: Performance: concatenate these together to speed up
                # TODO: Performance: split these into batches using map_fn to reduce the memory usage
                pred_true_open_heads = self._eval_padded_targets(heads=open_heads,
                                                                 rels=rels,
                                                                 tails=tails,
                                                                 masks=open_heads_mask,
                                                                 device=device)
                pred_true_open_tails = self._eval_padded_targets(heads=heads,
                                                                 rels=rels,
                                                                 tails=open_tails,
                                                                 masks=open_tails_mask,
                                                                 device=device)
                pred_true_closed_heads = self._eval_padded_targets(heads=closed_heads,
                                                                   rels=rels,
                                                                   tails=tails,
                                                                   masks=closed_heads_mask,
                                                                   device=device)
                pred_true_closed_tails = self._eval_padded_targets(heads=heads,
                                                                   rels=rels,
                                                                   tails=closed_tails,
                                                                   masks=closed_tails_mask,
                                                                   device=device)

                # Calculate the rank on the open and closed targets
                def __create_split_target_matrix(entities, n_splits):
                    if n_splits <= 1:
                        return tf.expand_dims(entities, axis=0), tf.shape(entities)[:1]

                    # split size of each partition besides the last one
                    target_split_size = tf.floor_div(tf.shape(entities)[0], n_splits, name='split_size')

                    split_len = tf.tile(tf.expand_dims(target_split_size, axis=0), [n_splits - 1])
                    last_split_len = tf.shape(entities)[:1] - (n_splits - 1) * target_split_size

                    tf.logging.info("split_len %s" % split_len.get_shape())
                    tf.logging.info("last_split_len %s" % last_split_len.get_shape())

                    target_split_len = tf.concat([split_len, last_split_len], axis=0)

                    tf.logging.info(
                        "target_split_size %s %s" % (target_split_size.get_shape(), target_split_size.dtype))

                    max_split_len = tf.reduce_max(target_split_len)

                    tf.logging.info("[%s] entities %s"
                                    "target_split_size %s "
                                    "target_split_len %s "
                                    "max_split_len %s" % (sys._getframe().f_code.co_name,
                                                          entities.get_shape(),
                                                          target_split_size.get_shape(),
                                                          target_split_len.get_shape(),
                                                          max_split_len.get_shape()))

                    entity_splits = tf.split(entities, target_split_len, axis=0)
                    entity_split_lens = tf.unstack(target_split_len, n_splits, axis=0)

                    tf.logging.info("[%s] entity_splits %s"
                                    "entity_split_lens %s " % (sys._getframe().f_code.co_name,
                                                               entity_splits[0].get_shape(),
                                                               entity_split_lens[0].get_shape()))

                    # Pad targets with -1
                    padded_entities = tf.pad(entities - 1,
                                             paddings=[[0, max_split_len * n_splits - tf.shape(entities)[0]]]) + 1
                    target_batches = tf.reshape(padded_entities, [n_splits, max_split_len])

                    tf.logging.info("[%s] target_batches %s" % (sys._getframe().f_code.co_name,
                                                                target_batches.get_shape()))

                    return target_batches, target_split_len

                # These two are shared among head and tail prediction

                split_open_targets, open_targets_len = __create_split_target_matrix(self.avoid_entities, n_splits)
                split_closed_targets, closed_targets_len = __create_split_target_matrix(self.closed_entities,
                                                                                        n_splits * 10)

                tf.logging.info("split_open_targets %s %s" % (split_open_targets, open_targets_len))
                tf.logging.info("split_closed_targets %s %s" % (split_closed_targets, closed_targets_len))

                def __evaluate_target_helper(x):
                    # The targets are padded targets,
                    # so we only want the true targets without the padding
                    targets, target_len = x

                    head_scores, tail_scores = self._eval_targets(heads=heads,
                                                                  rels=rels,
                                                                  tails=tails,
                                                                  targets=targets,
                                                                  device=device)

                    head_scores = tf.slice(head_scores, [0, 0], [-1, target_len])
                    tail_scores = tf.slice(tail_scores, [0, 0], [-1, target_len])

                    head_rank, head_eq = self._calculate_rank(head_scores, pred_scores)
                    tail_rank, tail_eq = self._calculate_rank(tail_scores, pred_scores)

                    return tf.stack([tf.reshape(x, [-1]) for x in [head_rank, head_eq, tail_rank, tail_eq]])

                # [n_splits, 4, batch_size]
                with tf.device(device):

                    open_ranks = tf.map_fn(__evaluate_target_helper,
                                           [split_open_targets, open_targets_len],
                                           dtype=tf.int32,
                                           parallel_iterations=1,
                                           back_prop=False,
                                           swap_memory=True,
                                           name='open_ranks')

                    # [n_splits, 4, batch_size]
                    closed_ranks = tf.map_fn(__evaluate_target_helper,
                                             [split_closed_targets, closed_targets_len],
                                             dtype=tf.int32,
                                             parallel_iterations=1,
                                             back_prop=False,
                                             swap_memory=True,
                                             name='closed_ranks')

                tf.logging.info("[%s] split_open_targets %s "
                                "open_ranks %s "
                                "split_closed_targets %s"
                                "closed_ranks %s " % (sys._getframe().f_code.co_name,
                                                      split_open_targets.get_shape(),
                                                      open_ranks.get_shape(),
                                                      split_closed_targets.get_shape(),
                                                      closed_ranks.get_shape()))

                # [batch_size, 4]
                all_open_heads_rank, all_open_heads_eq, all_open_tails_rank, all_open_tails_eq = tf.unstack(
                    tf.reduce_sum(open_ranks, axis=0), axis=0)
                all_closed_heads_rank, all_closed_heads_eq, all_closed_tails_rank, all_closed_tails_eq = tf.unstack(
                    tf.reduce_sum(closed_ranks, axis=0), axis=0)

                closed_tails_rank, closed_tails_eq = self._calculate_rank(pred_true_closed_tails, pred_scores)
                closed_heads_rank, cloesd_heads_eq = self._calculate_rank(pred_true_closed_heads, pred_scores)
                open_tails_rank, open_tails_eq = self._calculate_rank(pred_true_open_tails, pred_scores)
                open_heads_rank, open_tails_eq = self._calculate_rank(pred_true_open_heads, pred_scores)

                classic_head_rank = all_closed_heads_rank + 1
                classic_filtered_head_rank = classic_head_rank - closed_heads_rank
                classic_tail_rank = all_closed_tails_rank + 1
                classic_filtered_tail_rank = classic_tail_rank - closed_tails_rank

                new_head_rank = all_closed_heads_rank + all_open_heads_rank + 1
                new_filtered_head_rank = new_head_rank - closed_heads_rank - open_heads_rank
                new_tail_rank = all_closed_tails_rank + all_open_tails_rank + 1
                new_filtered_tail_rank = new_tail_rank - closed_tails_rank - open_tails_rank

                # Here we have the scores for this mini batch, then we need to distribute the scores to the accumulators according to their types
                # we use avoid_entities instead of the closed_entities because this should be significantly smaller
                head_in_train = tf.logical_not(self.entity_in_set_indicator(heads, self.avoid_entities))
                tail_in_train = tf.logical_not(self.entity_in_set_indicator(tails, self.avoid_entities))

                # Closed world
                # Two types, seen head -> seen tail AND seen tail -> seen head
                # This is the typical setting of the closed world KGC
                # seen head -> seen tail
                both_in_train_mask = tf.logical_and(head_in_train, tail_in_train, name='both_in_train_mask')
                closed_head_pred_tail = tf.boolean_mask(classic_tail_rank, both_in_train_mask)
                closed_filtered_head_pred_tail = tf.boolean_mask(classic_filtered_tail_rank, both_in_train_mask)
                closed_tail_pred_head = tf.boolean_mask(classic_head_rank, both_in_train_mask)
                closed_filtered_tail_pred_head = tf.boolean_mask(classic_filtered_head_rank, both_in_train_mask)

                # Open world
                # When predicting on unseen elements,
                #   the target entity candidates should be all entities including the unseen ones
                # When predicting on seen elements,
                #   the target entity candidates should be just closed entities
                both_not_in_train_mask = tf.logical_and(tf.logical_not(head_in_train),
                                                        tf.logical_not(tail_in_train))

                # unseen head -> unseen tail (target should be all entities in both open and closed world)
                open_unseen_head_to_unseen_tail = tf.boolean_mask(new_tail_rank, both_not_in_train_mask)
                open_filtered_unseen_head_to_unseen_tail = tf.boolean_mask(new_filtered_tail_rank,
                                                                           both_not_in_train_mask)

                # unseen tail -> unseen head
                open_unseen_tail_to_unseen_head = tf.boolean_mask(new_head_rank, both_not_in_train_mask)
                open_filtered_unseen_tail_to_unseen_head = tf.boolean_mask(new_filtered_head_rank,
                                                                           both_not_in_train_mask)

                head_not_in_tail_in_mask = tf.logical_and(tf.logical_not(head_in_train),
                                                          tail_in_train)
                # unseen head -> seen tail
                open_unseen_head_to_seen_tail = tf.boolean_mask(new_tail_rank, head_not_in_tail_in_mask)
                open_filtered_unseen_head_to_seen_tail = tf.boolean_mask(new_filtered_tail_rank,
                                                                         head_not_in_tail_in_mask)

                head_in_tail_not_in_mask = tf.logical_and(head_in_train, tf.logical_not(tail_in_train))
                # unseen tail -> seen head
                open_unseen_tail_to_seen_head = tf.boolean_mask(new_head_rank, head_in_tail_not_in_mask)
                open_filtered_unseen_tail_to_seen_head = tf.boolean_mask(new_filtered_head_rank,
                                                                         head_in_tail_not_in_mask)

                # seen head -> unseen tail
                open_seen_head_to_unseen_tail = tf.boolean_mask(new_tail_rank, head_in_tail_not_in_mask)
                open_filtered_seen_head_to_unseen_tail = tf.boolean_mask(new_filtered_tail_rank,
                                                                         head_in_tail_not_in_mask)
                # seen tail -> unseen head
                open_seen_tail_to_unseen_head = tf.boolean_mask(new_head_rank, head_not_in_tail_in_mask)
                open_filtered_seen_tail_to_unseen_head = tf.boolean_mask(new_filtered_head_rank,
                                                                         head_not_in_tail_in_mask)

                # Semi Open world
                # Unseen to seen, the target entities are the seen entities only
                # unseen head -> seen tail
                semi_open_unseen_head_to_seen_tail = tf.boolean_mask(classic_tail_rank, head_not_in_tail_in_mask)
                semi_open_filtered_unseen_head_to_seen_tail = tf.boolean_mask(classic_filtered_tail_rank,
                                                                              head_not_in_tail_in_mask)
                # unseen tail -> seen head
                semi_open_unseen_tail_to_seen_head = tf.boolean_mask(classic_head_rank, head_in_tail_not_in_mask)
                semi_open_filtered_unseen_tail_to_seen_head = tf.boolean_mask(classic_filtered_head_rank,
                                                                              head_in_tail_not_in_mask)

                with tf.name_scope('streaming_metrics'):
                    metric_data = [
                        [closed_head_pred_tail, 'closed_seen_head_to_seen_tail'],
                        [closed_filtered_head_pred_tail, 'closed_seen_head_to_seen_tail_filtered'],

                        [closed_tail_pred_head, 'closed_tail_pred_head'],
                        [closed_filtered_tail_pred_head, 'closed_seen_tail_to_seen_head_filtered'],

                        [tf.concat([closed_head_pred_tail, closed_tail_pred_head], axis=0), 'closed_mean_rank'],
                        [tf.concat([closed_filtered_head_pred_tail, closed_filtered_tail_pred_head], axis=0),
                         'closed_mean_rank_filtered'],

                        [open_unseen_head_to_unseen_tail, 'open_unseen_head_to_unseen_tail'],
                        [open_filtered_unseen_head_to_unseen_tail, 'open_unseen_head_to_unseen_tail_filtered'],

                        [open_unseen_tail_to_unseen_head, 'open_unseen_tail_to_unseen_head'],
                        [open_filtered_unseen_tail_to_unseen_head, 'open_unseen_tail_to_unseen_head_filtered'],

                        [open_unseen_head_to_seen_tail, 'open_unseen_head_to_seen_tail'],
                        [open_filtered_unseen_head_to_seen_tail, 'open_unseen_head_to_seen_tail_filtered'],

                        [open_unseen_tail_to_seen_head, 'open_unseen_tail_to_seen_head'],
                        [open_filtered_unseen_tail_to_seen_head, 'open_unseen_tail_to_seen_head_filtered'],

                        [open_seen_head_to_unseen_tail, 'open_seen_head_to_unseen_tail'],
                        [open_filtered_seen_head_to_unseen_tail, 'open_seen_head_to_unseen_tail_filtered'],

                        [open_seen_tail_to_unseen_head, 'open_seen_tail_to_unseen_head'],
                        [open_filtered_seen_tail_to_unseen_head, 'open_seen_tail_to_unseen_head_filtered'],

                        [semi_open_unseen_head_to_seen_tail, 'semi_open_unseen_head_to_seen_tail'],
                        [semi_open_filtered_unseen_head_to_seen_tail, 'semi_open_unseen_head_to_seen_tail_filtered'],

                        [semi_open_unseen_tail_to_seen_head, 'semi_open_unseen_tail_to_seen_head'],
                        [semi_open_filtered_unseen_tail_to_seen_head, 'semi_open_unseen_tail_to_seen_head_filtered'],

                        [tf.concat([open_unseen_head_to_unseen_tail,
                                    open_unseen_tail_to_unseen_head,
                                    open_unseen_head_to_seen_tail,
                                    open_unseen_tail_to_seen_head,
                                    open_seen_head_to_unseen_tail,
                                    open_seen_tail_to_unseen_head], axis=0),
                         'open_mean_rank'],
                        [tf.concat([open_filtered_unseen_head_to_unseen_tail,
                                    open_filtered_unseen_tail_to_unseen_head,
                                    open_filtered_unseen_head_to_seen_tail,
                                    open_filtered_unseen_tail_to_seen_head,
                                    open_filtered_seen_head_to_unseen_tail,
                                    open_filtered_seen_tail_to_unseen_head], axis=0),
                         'open_mean_rank_filtered'],

                        [tf.concat([semi_open_unseen_head_to_seen_tail,
                                    semi_open_unseen_tail_to_seen_head], axis=0),
                         'semi_open_mean_rank'],
                        [tf.concat([semi_open_filtered_unseen_head_to_seen_tail,
                                    semi_open_filtered_unseen_tail_to_seen_head], axis=0),
                         'semi_open_mean_rank_filtered']

                    ]
                    update_ops = list()

                    for v, k in metric_data:
                        m, u = metrics.streaming_mean(v, name=k)
                        tf.summary.scalar(k, m, collections=[self.EVAL_SUMMARY])
                        update_ops.append(u)

                tf.logging.info("[%s] classic_head_rank %s "
                                "classic_filtered_head_rank %s "
                                "classic_tail_rank %s "
                                "classic_filtered_tail_rank %s "
                                "new_head_rank %s "
                                "new_filtered_head_rank %s "
                                "new_tail_rank %s "
                                "new_filtered_tail_rank %s " % (sys._getframe().f_code.co_name,
                                                                classic_head_rank.get_shape(),
                                                                classic_filtered_head_rank.get_shape(),
                                                                classic_tail_rank.get_shape(),
                                                                classic_filtered_tail_rank.get_shape(),
                                                                new_head_rank.get_shape(),
                                                                new_filtered_head_rank.get_shape(),
                                                                new_tail_rank.get_shape(),
                                                                new_filtered_tail_rank.get_shape()))

                eval_result = {
                    'closed':
                        {
                            'head': {
                                'rank': classic_head_rank,
                                'frank': classic_filtered_head_rank,
                            },
                            'tail ': {
                                'rank': classic_tail_rank,
                                'frank': classic_filtered_tail_rank,
                            },
                            'head_same': all_closed_heads_eq,
                            'tail_same': all_closed_tails_eq,
                        },
                    'open':
                        {
                            'head': {
                                'rank': new_head_rank,
                                'frank': new_filtered_head_rank,
                            },
                            'tail': {
                                'rank': new_tail_rank,
                                'frank': new_filtered_tail_rank
                            },
                            'head_same': all_open_heads_eq,
                            'tail_same': all_open_tails_eq,
                        },
                }

                return ph_evaluation_triples, enqueue_triple, str_triple_batch, eval_result, tf.group(*update_ops)


def main(_):
    import os
    import sys
    tf.logging.set_verbosity(tf.logging.INFO)
    CHECKPOINT_DIR = sys.argv[1]
    dataset_dir = sys.argv[2]

    model = ContentModel(entity_file=os.path.join(dataset_dir, 'entities.txt'),
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
                         evaluation_closed_target_tail_file=os.path.join(dataset_dir, 'eval.tails.values.closed'),
                         evaluation_target_tail_key_file=os.path.join(dataset_dir, 'eval.tails.idx'),

                         evaluation_open_target_head_file=os.path.join(dataset_dir, 'eval.heads.values.open'),
                         evaluation_closed_target_head_file=os.path.join(dataset_dir, 'eval.heads.values.closed'),
                         evaluation_target_head_key_file=os.path.join(dataset_dir, 'eval.heads.idx'),

                         train_file=os.path.join(dataset_dir, 'train.txt'),

                         num_epoch=10,
                         word_oov=100,
                         word_embedding_size=200,
                         debug=True)

    model.create('/cpu:0')
    train_op, loss_op, merge_ops = model.train_ops(num_epoch=100, batch_size=200,
                                                   sampled_true=1, sampled_false=31,
                                                   devices=['/gpu:0', '/gpu:1', '/gpu:2'])
    ph_eval_triples, triple_enqueue_op, batch_data_op, batch_pred_score_op, metric_update_ops = model.auto_eval_ops(
        batch_size=1,
        n_splits=2000,
        device='/gpu:3')
    metric_reset_op = tf.variables_initializer([i for i in tf.local_variables() if 'streaming_metrics' in i.name])
    metric_merge_op = tf.summary.merge_all(model.EVAL_SUMMARY)

    config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    saver = tf.train.Saver(max_to_keep=3, var_list=tf.trainable_variables() + [model.global_step])

    # Evaluation dataset is here
    validation_data = load_triples(os.path.join(dataset_dir, 'valid.txt'))
    # ph, trip = model.auto_eval_ops(len(validation_data), batch_size=3)

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
        avoid_targets = intbitset(sess.run(model.avoid_entities).tolist())
        tf.logging.info("avoid targets %s" % avoid_targets)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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
                        _, loss, global_step, merged = sess.run([train_op, loss_op, model.global_step, merge_ops[0]])
                    train_writer.add_summary(merged, global_step)
                else:
                    _, loss, global_step = sess.run([train_op, loss_op, model.global_step])

                print("global_step %d loss %.4f" % (global_step, loss), end='\r')

                if global_step % 1000 == 0:
                    print("Saving model@%d" % global_step)
                    saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=global_step)
                    print("Saved.")

                    # feed evaluation data and reset metric scores
                    sess.run([triple_enqueue_op, metric_reset_op],
                             feed_dict={ph_eval_triples: validation_data})
                    s = 0
                    while s < len(validation_data):
                        sess.run([metric_update_ops])
                        s += min(len(validation_data) - s, 10)
                        print("evaluated %d elements" % s)
                    train_writer.add_summary(sess.run(metric_merge_op), global_step)
                    print("evaluation done")

        except tf.errors.OutOfRangeError:
            print("training done")
        finally:
            coord.request_stop()

        coord.join(threads)

        saver.save(sess, os.path.join(CHECKPOINT_DIR, "model.ckpt"), global_step=model.global_step)
        tf.logging.info("Model saved with %d global steps." % sess.run(model.global_step))


if __name__ == '__main__':
    tf.app.run()
