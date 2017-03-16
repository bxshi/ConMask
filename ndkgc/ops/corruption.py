import tensorflow as tf

from ndkgc.ops.sampling import single_negative_sampling

__corrupt_head = ['h', 'head']
__corrupt_tail = ['t', 'tail']


def _corrupt_single_entity_helper(triple: tf.Tensor,
                                  all_triples: tf.Tensor,
                                  corrupt_type: str, max_range: int,
                                  debug_corrupted_counter=None,
                                  name=None):
    """ Corrupt the entity by __sampling from [0, max_range] and not in the true target set.

    :param triple:
    :param all_triples:
    :param corrupt_type:
    :param max_range:
    :param name:
    :return: corrupted 1-d [h,r,t] triple
    """
    with tf.name_scope(name, "corrupt_an_entity", [triple, all_triples]):
        with tf.control_dependencies([debug_corrupted_counter.assign_add(1)]):
            h, r, t = tf.unstack(triple, name='unstack_triple', axis=0)

            # do relation match first to reduce future search space
            rel_mask = tf.equal(all_triples[:, 1], r, name='rel_mask')
            rel_matched_triples = tf.boolean_mask(all_triples, rel_mask, name='rel_matched_triples')

            if corrupt_type.lower() in __corrupt_head:
                # corrupt on head, so find all [?, r, t] pairs and get all positive head
                tail_mask = tf.equal(rel_matched_triples[:, 2], t)
                true_head_targets = tf.boolean_mask(rel_matched_triples[:, 0], tail_mask, name='true_head_targets')
                corrupted_head = tf.reshape(single_negative_sampling(true_head_targets, max_range), ())
                return tf.stack([corrupted_head, r, t], name='head_corrupted_triple', axis=0)
            elif corrupt_type.lower() in __corrupt_tail:
                head_mask = tf.equal(rel_matched_triples[:, 0], h)
                true_tail_targets = tf.boolean_mask(rel_matched_triples[:, 1], head_mask, name='true_tail_targets')
                corrupted_tail = tf.reshape(single_negative_sampling(true_tail_targets, max_range), ())
                return tf.stack([h, r, corrupted_tail], name='tail_corrupted_triple', axis=0)
            else:
                raise ValueError("Can only do head corruption or tail corruption, given %s" % corrupt_type)


def corrupt_single_relationship(triple: tf.Tensor,
                                all_triples: tf.Tensor,
                                max_range: int,
                                name=None):
    """ Corrupt the relationship by __sampling from [0, max_range]

    :param triple:
    :param all_triples:
    :param max_range:
    :param name:
    :return: corrupted 1-d [h,r,t] triple
    """
    with tf.name_scope(name, 'corrupt_single_relation', [triple, all_triples]):
        h, r, t = tf.unstack(triple, name='unstack_triple')

        head_mask = tf.equal(all_triples[:, 0], h, name='head_mask')
        head_matched_triples = tf.boolean_mask(all_triples[:, 1:], head_mask, name='head_matched_triples')

        tail_mask = tf.equal(head_matched_triples[:, 1], t, name='tail_mask')
        true_rels = tf.boolean_mask(head_matched_triples[:, 0], tail_mask)

        corrupted_rel = tf.reshape(single_negative_sampling(true_rels, max_range), ())

        return tf.stack([h, corrupted_rel, t], name='rel_corrupted_triple')


def corrupt_single_entity(triple: tf.Tensor, all_triples: tf.Tensor,
                          max_entity_id: int,
                          head_corrupt_prob=0.5,
                          debug_head_corrupted=None,
                          debug_tail_corrupted=None,
                          name=None):
    """ Randomly corrupt head or tail with prob `head_corrupt_prob`

    :param triple:
    :param all_triples:
    :param max_entity_id:
    :param head_corrupt_prob:
    :param name:
    :return: corrupted 1-d [h,r,t] triple
    """
    with tf.name_scope(name, "corrupt_single_entity", [triple, all_triples]):
        # if rand_val < 0.5, do head corruption, otherwise do tail corruption
        rand_val = tf.random_uniform(())
        corruption_cond = tf.less(rand_val, head_corrupt_prob, name='corruption_selector')
        entity_corrupted_triple = tf.cond(corruption_cond,
                                          lambda: _corrupt_single_entity_helper(triple, all_triples, 'h', max_entity_id,
                                                                                debug_head_corrupted, 'corrupt_head'),
                                          lambda: _corrupt_single_entity_helper(triple, all_triples, 't',
                                                                                max_entity_id,
                                                                                debug_tail_corrupted, 'corrupt_tail'))
        return entity_corrupted_triple
