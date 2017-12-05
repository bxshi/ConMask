import csv
from intbitset import intbitset

from ndkgc.models.content_model import ContentModel
from ndkgc.ops import *
from ndkgc.utils import *

# Skip these settings because they already exist in content_model.py
# tf.app.flags.DEFINE_boolean('eval', False, 'Run evaluation')
# tf.app.flags.DEFINE_integer('max_content', 256, 'Max content length')
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool("fix", False, 'Fix word embedding or not')
tf.app.flags.DEFINE_integer("layer", 3, 'Number of FCN layers')
tf.app.flags.DEFINE_integer("conv", 2, 'Number of Conv layers per FCN layer')
tf.app.flags.DEFINE_string('activation', 'sigmoid', 'activation function')
tf.app.flags.DEFINE_float("keep_prob", 0.5, 'keep probability')
tf.app.flags.DEFINE_float("lr", 1e-4, 'learning rate')
tf.app.flags.DEFINE_integer('neg', 4, "negative sampling")
tf.app.flags.DEFINE_integer('pos', 1, "positive sampling")
tf.app.flags.DEFINE_bool('filter', True, 'target filter')
tf.app.flags.DEFINE_integer('prev', 5, 'previous window size')
tf.app.flags.DEFINE_integer('neval', 500, 'run evaluation every *neval* iteration')
tf.app.flags.DEFINE_boolean("open", True, 'Open World')
FLAGS = tf.app.flags.FLAGS


class FCNModel(ContentModel):
    def __init__(self, **kwargs):
        super(FCNModel, self).__init__(**kwargs)

        self.fcn_scope = None
        with tf.variable_scope('fcn') as scp:
            self.fcn_scope = scp

    def _create_nontrainable_variables(self):
        super(FCNModel, self)._create_nontrainable_variables()

        with tf.variable_scope(self.non_trainable_scope):
            self.is_train = tf.Variable(True, trainable=False,
                                        collections=[self.NON_TRAINABLE],
                                        name='is_train')

    def lookup_entity_description_and_title(self, ents, name=None):
        return description_and_title_lookup(ents, self.entity_content, self.entity_content_len,
                                            self.entity_title, self.entity_title_len,
                                            self.vocab_table, self.word_embedding, self.PAD_const,
                                            name)

    def translate_triple(self, heads, tails, rels, device, reuse=True):
        with tf.name_scope("fcn_translate_triple"):
            # Keep using the averaged relationship
            # right now

            if len(heads.get_shape()) == 1:
                heads = tf.expand_dims(heads, axis=0)
            if len(tails.get_shape()) == 1:
                tails = tf.expand_dims(tails, axis=0)

            tf.logging.debug("[%s] heads: %s tails %s rels %s device %s" % (sys._getframe().f_code.co_name,
                                                                            heads.get_shape(),
                                                                            tails.get_shape(),
                                                                            rels.get_shape(),
                                                                            device))
            transformed_rels = self._transform_relation(rels,
                                                        reuse=reuse,
                                                        device=device)

            transformed_head_content, transformed_head_title = self._transform_head_entity(heads,
                                                                                           transformed_rels,
                                                                                           reuse=reuse,
                                                                                           device=device)
            transformed_tail_content, transformed_tail_title = self._transform_tail_entity(tails,
                                                                                           transformed_rels,
                                                                                           reuse=True,
                                                                                           device=device)

            tf.logging.debug("[%s] transformed_heads: %s "
                             "transformed_tails %s "
                             "transformed_rels %s" % (sys._getframe().f_code.co_name,
                                                      transformed_head_content.get_shape(),
                                                      transformed_tail_content.get_shape(),
                                                      transformed_rels.get_shape()))

            transformed_head_content_rel = self._combine_head_relation(transformed_head_content, transformed_rels,
                                                                       device=device)

            pred_scores = self._predict(transformed_head_content_rel,
                                        transformed_head_title,
                                        transformed_tail_content,
                                        transformed_tail_title,
                                        device=device,
                                        reuse=reuse)

            tf.logging.debug("pred_scores %s" % (pred_scores))
            return pred_scores

    def _predict(self, head_content, head_title, tail_content, tail_title, device='/cpu:0', reuse=True, name=None):
        with tf.name_scope(name, 'predict', [head_content, head_title, tail_content, tail_title]):
            with tf.variable_scope(self.pred_scope, reuse=reuse):
                with tf.device(device):
                    head_content, head_title, tail_content, tail_title = [normalized_embedding(x) for x in
                                                                          [head_content, head_title, tail_content,
                                                                           tail_title]]

                    def predict_helper(a, b):
                        return tf.reduce_sum(a * b, axis=-1)

                    # similarity score between content
                    pred_score = predict_helper(head_content, tail_content)
                    # similarity scores between head content and tail title
                    # head_content_tail_title_sim = predict_helper(head_content, tail_title)
                    # similarity scores between tail content and head title
                    # tail_content_head_title_sim = predict_helper(tail_content, head_title)
                    # similarity between two titles
                    # title_sim = predict_helper(head_title, tail_title)

                    # sim_scores = tf.stack([content_sim, head_content_tail_title_sim,
                    #                        tail_content_head_title_sim, title_sim], axis=0)
                    # tf.logging.info("stacked sim_scores %s" % sim_scores.get_shape())
                    #
                    # pred_score = tf.check_numerics(
                    #     tf.reduce_sum(sim_scores * self.predict_weight, axis=0, name='orig_pred_score'), '__predict')

                    # Rescale logits by minus the max score, this is used to deal with NAN gradient when
                    # using activation function such as ReLu
                    # This is not the cause of nan
                    # pred_score = pred_score - tf.reduce_max(pred_score, axis=1, name='stable_pred_score', keep_dims=True)

                    return pred_score

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
                return tf.check_numerics(transformed_rels, 'transform_relation')

    def __transform_entity(self, ents, transformed_rels, reuse=True, device='/cpu:0', name=None):
        """ This is the transformation function for both head and tail entities

        :param ents:
        :param transformed_rels:
        :param reuse:
        :param device:
        :param name:
        :return:
        """

        varlist = [ents, transformed_rels, self.word_embedding, self.vocab_table, self.entity_content,
                   self.entity_content_len, self.entity_title, self.entity_title_len, self.PAD_const,
                   self.is_train]

        with tf.name_scope(name, 'transform_entity', varlist):
            (ent_content, ent_content_len), (ent_title, ent_title_len) = description_and_title_lookup(ents,
                                                                                                      self.entity_content,
                                                                                                      self.entity_content_len,
                                                                                                      self.entity_title,
                                                                                                      self.entity_title_len,
                                                                                                      self.vocab_table,
                                                                                                      self.word_embedding,
                                                                                                      self.PAD_const)

            pad_word_embedding = tf.check_numerics(
                self.word_embedding[tf.cast(self.vocab_table.lookup(self.PAD_const), tf.int32), :],
                'pad_word_embedding')

            with tf.device(device):
                masked_ent_content = tf.check_numerics(
                    mask_content_embedding(ent_content, transformed_rels, name='masked_content'), 'masked_ent_content')
                masked_ent_content = tf.check_numerics(masked_ent_content, 'masked_ent_content')
                # Do FCN here

                extracted_ent_content = tf.check_numerics(extract_embedding_by_dkrl(masked_ent_content,
                                                                                    filters=self.word_embedding_size,
                                                                                    variable_scope=self.fcn_scope,
                                                                                    reuse=reuse),
                                                          'extracted_ent_content')

                avg_title = tf.check_numerics(
                    avg_content(ent_title, ent_title_len, pad_word_embedding, name='avg_title'), 'avg_title')

                # TODO: add a semantic background embedding extracted from content,
                # this could be a simple word averaging or another FCN without masking or LSTM

                return extracted_ent_content, avg_title

    def _transform_head_entity(self, heads, transformed_rels, reuse=True, device='/cpu:0', name=None):
        """
        This is used to extract entity description and titles.
        :param heads: [?, ?] <- due to evaluation, sometimes heads will be (1, ?) but transformed_rels will still be (batch, word_dim)
        :param rel_embedding: [?, word_dim]
        :param rel_embedding_len: [?, 1]
        :param reuse:
        :param device:
        :param name:
        :return:
        """
        return self.__transform_entity(heads, transformed_rels, reuse, device, name='head_entity')

    def _transform_tail_entity(self, tails, transformed_rels, reuse=True, device='/cpu:0', name=None):
        return self.__transform_entity(tails, transformed_rels, reuse, device, name='tail_entity')

    def manual_eval_head_ops_v2(self, device='/cpu:0'):

        with tf.name_scope("manual_evaluation_v2"):
            with tf.device(device):
                ph_tail_rel = tf.placeholder(tf.string, [1, 2], name='ph_tail_rel')
                ph_eval_targets = tf.placeholder(tf.string, [1, None], name='ph_eval_targets')
                ph_true_target_idx = tf.placeholder(tf.int32, [None], name='ph_true_target_idx')
                # indices of true targets in the evaluation set
                ph_test_target_idx = tf.placeholder(tf.int32, [None], name='ph_test_target_idx')

                ph_target_size = tf.placeholder(tf.int32, (), name='ph_target_size')

                str_tails, str_rels = tf.unstack(ph_tail_rel, axis=1)
                tails = self.entity_table.lookup(str_tails)
                rels = self.relation_table.lookup(str_rels)

                pre_computed_head_queue = tf.FIFOQueue(1000000, dtypes=[tf.float32, tf.float32],
                                                       shapes=[[self.word_embedding_size], [self.word_embedding_size]],
                                                       name='tail_queue')

                eval_heads = self.entity_table.lookup(ph_eval_targets)
                computed_rels = self._transform_relation(rels, reuse=True, device=device)
                computed_content_heads, computed_title_heads = [tf.squeeze(x, axis=0) for x in
                                                                self._transform_head_entity(eval_heads, computed_rels,
                                                                                            reuse=True, device=device)]

                pre_computed_heads = pre_computed_head_queue.enqueue_many(
                    [computed_content_heads, computed_title_heads])

                dequeue_op = pre_computed_head_queue.dequeue_many(ph_target_size)
                head_content_embed, head_title_embeds = [tf.expand_dims(x, axis=0) for x in dequeue_op]

                with tf.control_dependencies(dequeue_op):
                    re_enqueue = pre_computed_head_queue.enqueue_many(dequeue_op)

                computed_content_tails, computed_title_tails = self._transform_tail_entity(tails, computed_rels,
                                                                                           reuse=True, device=device)

                computed_content_head_rels = self._combine_head_relation(head_content_embed, computed_rels,
                                                                         device=device)
                pred_scores = tf.reshape(self._predict(computed_content_head_rels,
                                                       head_title_embeds,
                                                       computed_content_tails,
                                                       computed_title_tails,
                                                       device=device,
                                                       reuse=True), [-1, 1])

                ranks, rr = self.eval_helper(pred_scores, ph_test_target_idx, ph_true_target_idx)
                rand_ranks, rand_rr = self.eval_helper(
                    tf.random_uniform(tf.shape(pred_scores), minval=-1, maxval=1, dtype=tf.float32),
                    ph_test_target_idx, ph_true_target_idx)

                top_10_score, top_10 = tf.nn.top_k(tf.reshape(pred_scores, [-1]), k=10)

                return ph_tail_rel, ph_eval_targets, ph_target_size, pre_computed_head_queue.size(), \
                       ph_true_target_idx, ph_test_target_idx, \
                       pre_computed_heads, re_enqueue, dequeue_op, ranks, rr, rand_ranks, rand_rr, pred_scores, top_10, top_10_score

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

                # First, convert string to indices
                str_heads, str_rels = tf.unstack(ph_head_rel, axis=1)
                heads = self.entity_table.lookup(str_heads)
                rels = self.relation_table.lookup(str_rels)

                # A temporary queue for precomputed tails
                pre_computed_tail_queue = tf.FIFOQueue(1000000, dtypes=[tf.float32, tf.float32],
                                                       shapes=[[self.word_embedding_size], [self.word_embedding_size]],
                                                       # This may needs to be change later
                                                       name='tail_queue')

                # Convert string targets to numerical ids
                eval_tails = self.entity_table.lookup(ph_eval_targets)
                # computed tails [1, ?, word_dim]
                computed_rels = self._transform_relation(rels, reuse=True, device=device)
                computed_content_tails, computed_title_tails = [tf.squeeze(x, axis=0) for x in
                                                                self._transform_tail_entity(eval_tails, computed_rels,
                                                                                            reuse=True, device=device)]

                # put pre-computed tails into target queue
                # Call this to pre-compute tails for a certain relationship
                pre_compute_tails = pre_computed_tail_queue.enqueue_many([computed_content_tails, computed_title_tails])

                # get pre-computed tails from target queue
                dequeue_op = pre_computed_tail_queue.dequeue_many(ph_target_size)
                tail_content_embeds, tail_title_embeds = [tf.expand_dims(x, axis=0) for x in dequeue_op]
                # tf.logging.info("tail_embeds shape %s" % tail_embeds.get_shape())
                # Put tails back into the queue (this will run after tails are dequeued)
                with tf.control_dependencies(dequeue_op):
                    re_enqueue = pre_computed_tail_queue.enqueue_many(dequeue_op)

                # Calculate heads and tails
                computed_content_heads, computd_title_heads = self._transform_head_entity(heads, computed_rels,
                                                                                          reuse=True, device=device)
                computed_content_heads_rels = self._combine_head_relation(computed_content_heads, computed_rels,
                                                                          device=device)
                # This is the score of all the targets given a single partial triple
                pred_scores = tf.reshape(self._predict(computed_content_heads_rels,
                                                       computd_title_heads,
                                                       tail_content_embeds,
                                                       tail_title_embeds,
                                                       device=device,
                                                       reuse=True), [-1, 1])

                tf.logging.debug("eval pred_scores %s" % pred_scores.get_shape())

                ranks, rr = self.eval_helper(pred_scores, ph_test_target_idx, ph_true_target_idx)

                rand_ranks, rand_rr = self.eval_helper(
                    tf.random_uniform(tf.shape(pred_scores), minval=-1, maxval=1, dtype=tf.float32),
                    ph_test_target_idx, ph_true_target_idx)

                top_10_score, top_10 = tf.nn.top_k(tf.reshape(pred_scores, [-1]), k=10)

                return ph_head_rel, ph_eval_targets, ph_target_size, pre_computed_tail_queue.size(), \
                       ph_true_target_idx, ph_test_target_idx, \
                       pre_compute_tails, re_enqueue, dequeue_op, ranks, rr, rand_ranks, rand_rr, pred_scores, top_10, top_10_score


def main(_):
    import os
    import sys
    tf.logging.set_verbosity(tf.logging.INFO)
    CHECKPOINT_DIR = sys.argv[1]
    dataset_dir = sys.argv[2]

    is_train = not FLAGS.eval

    model = FCNModel(entity_file=os.path.join(dataset_dir, 'entities.txt'),
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
        '/gpu:3')

    ph_tail_rel_, ph_eval_targets_, ph_target_size_, q_size_, ph_true_target_idx_, \
    ph_test_target_idx_, pre_compute_heads_, re_enqueue_, dequeue_op_, ranks_, rr_, rand_ranks_, rand_rr_, _, top_10_id_, top_10_score_ = model.manual_eval_head_ops_v2(
        '/gpu:3')

    metric_merge_op = tf.summary.merge_all(model.EVAL_SUMMARY)

    EVAL_BATCH = 2000
    # ph_eval_triples, triple_enqueue_op, batch_data_op, batch_pred_score_op, metric_update_ops = model.auto_eval_ops(
    #     batch_size=EVAL_BATCH,
    #     n_splits=EVAL_SPLITS,
    #     device='/gpu:3')
    # metric_reset_op = tf.variables_initializer([i for i in tf.local_variables() if 'streaming_metrics' in i.name])
    # metric_merge_op = tf.summary.merge_all(model.EVAL_SUMMARY)

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
        avoid_targets = intbitset(sess.run(model.avoid_entities).tolist())
        tf.logging.info("avoid targets %s" % avoid_targets)

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

            sess.run(model.is_train.assign(False))
            print(sess.run(model.is_train))

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
            sess.run(model.is_train.assign(True))
            return np.mean(all_ranks), all_ranks, np.mean(all_rr), np.mean(all_multi_rr)

        def tail_eval_helper(is_test=True, target_filter=True, eval_subset=set(), closed=False):
            tf.logging.info("target_filter %s" % target_filter)
            # Set mode to evaluation
            sess.run(model.is_train.assign(False))
            print(sess.run(model.is_train))
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
            sess.run(model.is_train.assign(True))
            return np.mean(all_ranks), all_ranks, np.mean(all_rr), np.mean(all_multi_rr)

        if is_train:
            # make sure this is in the training model.
            sess.run(model.is_train.assign(True))
            train_writer = tf.summary.FileWriter(CHECKPOINT_DIR, sess.graph, flush_secs=60)
            try:
                global_step = sess.run(model.global_step)
                while not coord.should_stop():

                    if is_train:
                        if global_step % 10 == 0:
                            if global_step % 500 == 0:
                                _, loss, global_step, merged, merged_slow = sess.run(
                                    [train_op, loss_op, model.global_step, merge_ops[0], merge_ops[1]])
                                train_writer.add_summary(merged_slow, global_step)
                            else:
                                _, loss, global_step, merged = sess.run(
                                    [train_op, loss_op, model.global_step, merge_ops[0]])
                            train_writer.add_summary(merged, global_step)
                        else:
                            _, loss, global_step = sess.run([train_op, loss_op, model.global_step])

                        print("global_step %d loss %.4f" % (global_step, loss), end='\r')

                        if global_step % FLAGS.neval == 0:
                            print("Saving model@%d" % global_step)
                            saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=global_step)
                            print("Saved.")

                            mr, r, mrr, triple_mrr = tail_eval_helper(FLAGS.force_eval, target_filter=FLAGS.filter,
                                                                      closed=not FLAGS.open)

                            mr, r, mrr, triple_mrr = head_eval_helper(FLAGS.force_eval, target_filter=FLAGS.filter,
                                                                      closed=not FLAGS.open)

                            model.mean_rank.load(mr, sess)
                            model.mrr.load(mrr, sess)
                            model.triple_mrr.load(triple_mrr)

                            sess.run(metric_merge_op, feed_dict={model.rank_list: r})

            except tf.errors.OutOfRangeError:
                print("training done")
            finally:
                coord.request_stop()

            coord.join(threads)

        else:
            tail_eval_helper(target_filter=FLAGS.filter,
                             closed=not FLAGS.open)

            head_eval_helper(target_filter=FLAGS.filter, closed=not FLAGS.open)


if __name__ == '__main__':
    tf.app.run()
