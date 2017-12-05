import tensorflow as tf


class RandomModel(object):
    def manual_eval_ops(self, device='/cpu:0'):
        """ This is the baseline random model, this takes all the targets,
        randomly assign values to it and then report the result.

        :param device:
        :return:
        """

        with tf.name_scope("namual_evaluation"):
            with tf.device('/cpu:0'):
                # head rel pair to evaluate
                ph_head_rel = tf.placeholder(tf.string, [1, 2], name='ph_head_rel')
                # tail targets to evaluate
                ph_eval_targets = tf.placeholder(tf.string, [1, None], name='ph_eval_targets')
                # indices of true tail targets in ph_eval_targets. Mask these when calculating filtered mean rank
                ph_true_target_idx = tf.placeholder(tf.int32, [None], name='ph_true_target_idx')
                # indices of true targets in the evaluation set, we will return the ranks of these targets
                ph_test_target_idx = tf.placeholder(tf.int32, [None], name='ph_test_target_idx')

                # We put random numbers into the pred_scores_queue
                pred_scores_queue = tf.FIFOQueue(1000000, dtypes=tf.float32, shapes=[[1]], name='pred_scorse_queue')
