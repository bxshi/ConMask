from intbitset import intbitset

from ndkgc.models.content_model import ContentModel
from ndkgc.ops import *
from ndkgc.utils import *


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

        self.predict_weight = tf.Variable([[[1.]]] * 4, trainable=True, name='predict_weight')
        tf.summary.histogram(self.predict_weight.name, self.predict_weight, collections=[self.TRAIN_SUMMARY_SLOW])

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

            tf.logging.info("[%s] heads: %s tails %s rels %s device %s" % (sys._getframe().f_code.co_name,
                                                                           heads.get_shape(),
                                                                           tails.get_shape(),
                                                                           rels.get_shape(),
                                                                           device))
            transformed_rels = self.__transform_relation(rels,
                                                         reuse=reuse,
                                                         device=device)

            transformed_head_content, transformed_head_title = self.__transform_head_entity(heads,
                                                                                            transformed_rels,
                                                                                            reuse=reuse,
                                                                                            device=device)
            transformed_tail_content, transformed_tail_title = self.__transform_tail_entity(tails,
                                                                                            transformed_rels,
                                                                                            reuse=True,
                                                                                            device=device)

            tf.logging.info("[%s] transformed_heads: %s "
                            "transformed_tails %s "
                            "transformed_rels %s" % (sys._getframe().f_code.co_name,
                                                     transformed_head_content.get_shape(),
                                                     transformed_tail_content.get_shape(),
                                                     transformed_rels.get_shape()))

            pred_scores = self.__predict(transformed_head_content,
                                         transformed_head_title,
                                         transformed_tail_content,
                                         transformed_tail_title,
                                         device=device,
                                         reuse=reuse)

            tf.logging.info("pred_scores %s" % (pred_scores))
            return pred_scores

    def __predict(self, head_content, head_title, tail_content, tail_title, device='/cpu:0', reuse=True, name=None):
        with tf.name_scope(name, 'predict', [head_content, head_title, tail_content, tail_title, self.predict_weight]):
            with tf.variable_scope(self.pred_scope, reuse=reuse):
                with tf.device(device):
                    head_content, head_title, tail_content, tail_title = [normalized_embedding(x) for x in
                                                                          [head_content, head_title, tail_content,
                                                                           tail_title]]

                    def predict_helper(a, b):
                        return tf.reduce_sum(a * b, axis=-1)

                    # similarity score between content
                    content_sim = predict_helper(head_content, tail_content)
                    # similarity scores between head content and tail title
                    head_content_tail_title_sim = predict_helper(head_content, tail_title)
                    # similarity scores between tail content and head title
                    tail_content_head_title_sim = predict_helper(tail_content, head_title)
                    # similarity between two titles
                    title_sim = predict_helper(head_title, tail_title)

                    sim_scores = tf.stack([content_sim, head_content_tail_title_sim,
                                           tail_content_head_title_sim, title_sim], axis=0)
                    tf.logging.info("stacked sim_scores %s" % sim_scores.get_shape())
                    tf.logging.info("sim_scores w %s" % self.predict_weight.get_shape())

                    pred_score = tf.check_numerics(
                        tf.reduce_sum(sim_scores * self.predict_weight, axis=0, name='orig_pred_score'), '__predict')

                    # Rescale logits by minus the max score, this is used to deal with NAN gradient when
                    # using activation function such as ReLu
                    # This is not the cause of nan
                    # pred_score = pred_score - tf.reduce_max(pred_score, axis=1, name='stable_pred_score', keep_dims=True)

                    return pred_score

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

                extracted_ent_content = tf.check_numerics(extract_embedding_by_fcn(masked_ent_content,
                                                                                   conv_per_layer=2,
                                                                                   filters=self.word_embedding_size,
                                                                                   n_layer=3,
                                                                                   is_train=self.is_train,
                                                                                   window_size=3,
                                                                                   keep_prob=0.85,
                                                                                   variable_scope=self.fcn_scope,
                                                                                   reuse=reuse),
                                                          'extracted_ent_content')

                avg_title = tf.check_numerics(
                    avg_content(ent_title, ent_title_len, pad_word_embedding, name='avg_title'), 'avg_title')

                return extracted_ent_content, avg_title

    def __transform_head_entity(self, heads, transformed_rels, reuse=True, device='/cpu:0', name=None):
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

    def __transform_tail_entity(self, tails, transformed_rels, reuse=True, device='/cpu:0', name=None):
        return self.__transform_entity(tails, transformed_rels, reuse, device, name='tail_entity')


def main(_):
    import os
    import sys
    tf.logging.set_verbosity(tf.logging.INFO)
    CHECKPOINT_DIR = sys.argv[1]
    dataset_dir = sys.argv[2]

    is_train = not(len(sys.argv) == 4 and sys.argv[3] == 'eval')

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
                     debug=True)

    model.create('/cpu:0')

    if is_train:
        train_op, loss_op, merge_ops = model.train_ops(lr=1e-4, num_epoch=100, batch_size=200,
                                                       sampled_true=1, sampled_false=4,
                                                       devices=['/gpu:0', '/gpu:1', '/gpu:2'])
    else:
        tf.logging.info("Evaluate mode")
        ph_head_rel, ph_eval_targets, ph_true_target_idx, \
        ph_test_target_idx, eval_op, ranks, rr, rand_ranks, rand_rr = model.manual_eval_ops('/gpu:3')

    EVAL_BATCH = 500
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
        if is_train:
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

                    if global_step % 1000 == 0:
                        print("Saving model@%d" % global_step)
                        saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=global_step)
                        print("Saved.")

                        # feed evaluation data and reset metric scores
                        # sess.run([triple_enqueue_op, metric_reset_op, model.is_train.assign(False)],
                        #          feed_dict={ph_eval_triples: validation_data})
                        # s = 0
                        # while s < len(validation_data):
                        #     sess.run([metric_update_ops])
                        #     s += min(len(validation_data) - s, EVAL_BATCH)
                        #     print("evaluated %d elements" % s)
                        # train_writer.add_summary(sess.run(metric_merge_op), global_step)
                        # print("evaluation done")
                        # sess.run(model.is_train.assign(True))
                else:

                    # Set mode to evaluation
                    sess.run(model.is_train.assign(True))
                    print(sess.run(model.is_train))
                    # ph_head_rel, ph_eval_targets, ph_true_target_idx, ph_test_target_idx, ranks, rr

                    # First load evaluation data
                    evaluation_data = load_manual_evaluation_file(os.path.join(dataset_dir, 'test.txt'),
                                                                  os.path.join(dataset_dir, 'avoid_entities.txt'))

                    relation_specific_targets = load_relation_specific_targets(
                        os.path.join(dataset_dir, 'train.heads.idx'),
                        os.path.join(dataset_dir, 'relations.txt'))
                    filtered_targets = load_filtered_targets(os.path.join(dataset_dir, 'eval.tails.idx'),
                                                             os.path.join(dataset_dir, 'eval.tails.values.closed'))

                    all_ranks = list()
                    all_rr = list()
                    all_multi_rr = list()

                    random_ranks = list()
                    random_rr = list()
                    random_multi_rr = list()

                    # Randomly assign some values to the targets, and then run the evaluation

                    missed = 0
                    for c, (head_rel_str, eval_true_targets_set) in enumerate(evaluation_data.items()):
                        head_str, rel_str = head_rel_str.split('\t')
                        # [1, 2]
                        head_rel = [[head_str, rel_str]]

                        if rel_str not in relation_specific_targets:
                            tf.logging.warning("Relation %s does not have any valid targets!" % rel_str)
                        elif len(eval_true_targets_set) == 0:
                            tf.logging.warning("%s->%s->? has 0 targets" % (head_str, rel_str))
                        else:
                            # targets we will evaluate in this batch, all other targets will be
                            # ignored and consider as irrelevant
                            eval_targets_set = relation_specific_targets[rel_str]

                            # TODO: Performance: If eval_targets_set is too large, break it into small batches instead

                            # Find true targets (in train/valid/test) of the given head relation in the evaluation set and skip all others
                            true_targets = set(filtered_targets[head_rel_str]).intersection(eval_targets_set)
                            # for v in true_targets:
                            #     assert v in eval_targets_set

                            # find true evaluation targets in the test set that are in this set
                            eval_true_targets = set.intersection(eval_targets_set, eval_true_targets_set)
                            # for v in eval_true_targets:
                            #     assert v in eval_targets_set

                            # how many true targets we missed/filtered out
                            missed += len(eval_true_targets_set) - len(eval_true_targets)

                            if len(eval_true_targets) == 0:
                                tf.logging.debug("find %s/%s true evaluation targets in the evaluation set" % (
                                    len(eval_true_targets),
                                    len(eval_true_targets_set)))
                            else:
                                eval_targets = list(eval_targets_set)

                                test_target_idx = sorted([eval_targets.index(x) for x in eval_true_targets])
                                true_target_idx = sorted([eval_targets.index(x) for x in true_targets])

                                # filter target set always is a superset of evaluation target set
                                assert len(true_target_idx) >= len(test_target_idx)

                                # print("head rel", head_rel_str)
                                # print("true_targets: ", [eval_targets[x] for x in true_target_idx])
                                # print("eval_true_targets: ", [eval_targets[x] for x in test_target_idx])
                                # print("eval_targets:", eval_targets)

                                # First calculate the scores and put then into a queue for future analysis
                                start = 0
                                while start < len(eval_targets):
                                    end = min(start + EVAL_BATCH, len(eval_targets))
                                    sess.run(eval_op, feed_dict={ph_head_rel: head_rel,
                                                                 ph_eval_targets: [eval_targets[start:end]]})
                                    start = end

                                eval_dict = {ph_head_rel: head_rel,
                                             ph_true_target_idx: true_target_idx,
                                             ph_test_target_idx: test_target_idx}
                                _rand_ranks, _rand_rr = sess.run([rand_ranks, rand_rr], feed_dict=eval_dict)

                                _ranks, _rr = sess.run([ranks, rr], feed_dict=eval_dict)
                                # print(len(eval_true_targets), " target ranks", _ranks)
                                # print("scores", _scores)
                                # print("masked scores", _masked_scores.shape)
                                # print("pred scores", _pred_scores.shape)
                                all_ranks.extend([float(x) for x in _ranks])
                                all_rr.append(_rr)
                                all_multi_rr.extend([1.0 / float(x) for x in _ranks])

                                random_ranks.extend([float(x) for x in _rand_ranks])
                                random_rr.append(_rand_rr)
                                random_multi_rr.extend([1.0 / float(x) for x in _rand_ranks])

                                print("%d/%d %d "
                                      "MR %.4f (%.4f) "
                                      "MRR(per head,rel) %.4f (%.4f) "
                                      "MRR(per tail) %.4f (%.4f) missed %d" % (
                                          c, len(evaluation_data), len(all_ranks),
                                          np.mean(all_ranks), np.mean(random_ranks),
                                          np.mean(all_rr), np.mean(random_rr),
                                          np.mean(all_multi_rr), np.mean(random_multi_rr),
                                          missed), end='\r')
                                # exit(0)
                    print("")
                    exit(0)

        except tf.errors.OutOfRangeError:
            print("training done")
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
