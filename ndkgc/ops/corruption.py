import sys

import tensorflow as tf
import tensorflow.contrib.lookup as lookup

from ndkgc.ops.sampling import single_negative_sampling, multiple_negative_sampling

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


def get_target_entities(entity: tf.Tensor, relation: tf.Tensor,
                        targets_lookup_table: lookup.HashTable,
                        entity_table: lookup.HashTable,
                        targets: tf.Tensor,
                        name=None):
    """ Change this to return a SparseTensor

    :param entity: entity must be a 1-D vector with one element?
    :param relation:
    :param targets_lookup_table:
    :param entity_table:
    :param targets:
    :param ignore_invalid_indices
    :param name:
    :return:
    """
    with tf.name_scope(name, 'get_target_entities', [entity, relation, targets_lookup_table, targets]):
        lookup_key = tf.add(entity, tf.add('\t', relation), name='ent_rel_lookup_key')
        tf.logging.debug("lookup key shape %s" % lookup_key.get_shape())
        # Explicitly convert the idx to a scalar and extract the targets
        # [?]
        target_entities_lookup_id = targets_lookup_table.lookup(lookup_key)
        # CHECK IF WE HAVE -1 HERE, if so the error will be have a -2 that is out of the range
        target_entities_lookup_id = tf.where(tf.equal(target_entities_lookup_id, -1),
                                             target_entities_lookup_id - 1,
                                             target_entities_lookup_id)
        # sparseTensor
        str_targets = tf.string_split(tf.nn.embedding_lookup(targets, target_entities_lookup_id), delimiter=' ')

        int_target_vals = tf.cond(tf.equal(tf.shape(str_targets.values)[0], 0),
                                  lambda: tf.constant([], dtype=tf.int32, name='empty_target_entities'),
                                  lambda: tf.cast(entity_table.lookup(str_targets.values), tf.int32,
                                                  name='target_entities'))

        int_targets = tf.SparseTensor(str_targets.indices, int_target_vals, str_targets.dense_shape)
        return int_targets


def corrupt_single_entity_w_multiple_targets(triple: tf.Tensor,
                                             target_tail_table: lookup.HashTable,
                                             target_head_table: lookup.HashTable,
                                             target_tails: tf.Tensor,
                                             target_heads: tf.Tensor,
                                             avoid_entities: tf.Tensor,
                                             entity_table: lookup.HashTable,
                                             relation_table: lookup.HashTable,
                                             max_entity_id,
                                             num_true: 1,
                                             num_false: 5,
                                             head_corrupt_prob=0.5,
                                             name=None):
    """ Generate a set of training triples using a correct one.

    Input triple is a string triple, and the output elements are all numerical ids

    :param triple: A given triple in shape of [1, 3]
    :param target_tail_table: A HashTable, key is "\t".join([head, rel]) value is a
                                numerical id pointing to the row in target_tails.
    :param target_head_table: A HashTable, key is "\t".join([tail, rel]) value is a
                                numerical id pointing to the row in target_heads.
    :param target_tails: A TF matrix, each row is a string in format of " ".join(targets).
                         Targets is a list of target name strings.
    :param target_heads: A TF matrix, each row is a string in format of " ".join(targets).
                         Targets is a list of target name strings.
    :param avoid_entities: A TF tensor. This is a 1-D numerical id vector of all entities
                            that are not seen during training.
    :param entity_table: A HashTable, key is the entity string name,
                            value is its numerical id.
    :param relation_table: A HashTable, key is the relationship string name,
                            value is its numerical id.
    :param max_entity_id: The largest entity id in entity_table. Should be number
                            of entities - 1.
    :param num_true: Number of true targets we will sample.
    :param num_false: Number of false targets we will sample.
    :param head_corrupt_prob: The probability of head entity been corrupted. Default is 0.5
    :param name: Optional name of this op.
    :return:
        A tuple with 5 elements:
            A boolean tf scalar of whether head entity is corrupted;
            Numerical id of uncorrupted entity (can be head or tail);
            Numerical id of the relationship;
            [num_true] numerical id of sampled true targets;
            [num_false] numerical id of sampled false targets

    """
    with tf.name_scope(name, 'corrupt_single_entity_w_multiple_targets',
                       [triple, target_tail_table, target_head_table, target_tails, target_heads]):
        # Prob of head/tail corruption
        corrupt_rand = tf.random_uniform((), name='corrupt_rand')
        corrupt_head = tf.less(corrupt_rand, head_corrupt_prob, name='corrupt_cond')

        head, rel, tail = tf.unstack(triple, axis=-1)
        # First get the targets based on the selection result
        # Here we only use the values of a sparse tensor because the corruption is on a single triple
        true_targets = tf.cond(corrupt_head,
                               lambda: get_target_entities(tail, rel, target_head_table,
                                                           entity_table, target_heads,
                                                           name='get_target_heads').values,
                               lambda: get_target_entities(head, rel, target_tail_table,
                                                           entity_table, target_tails,
                                                           name='get_target_tails').values,
                               name='true_targets')

        # Then select num_true positive targets from true_targets
        selected_true_idx = tf.random_uniform([num_true], minval=0,
                                              maxval=tf.shape(true_targets)[0], dtype=tf.int32)

        tf.logging.debug("[%s] selected true idx %s" % (sys._getframe().f_code.co_name,
                                                        selected_true_idx.get_shape()))

        # Get the actual entity string name of sampled true targets
        sampled_true = tf.nn.embedding_lookup(true_targets, selected_true_idx, name='sampled_true_targets')

        # Now sample num_false negative targets that does not overlap with true_targets and avoid_entities
        avoid_entities = tf.cond(tf.equal(tf.rank(avoid_entities), 0),
                                 lambda: tf.expand_dims(avoid_entities, axis=0),
                                 lambda: tf.reshape(avoid_entities, [-1]))

        # when doing the negative sampling, ignore all avoid entities and true targets
        skip_targets = tf.concat([true_targets, avoid_entities], axis=-1)
        sampled_false = multiple_negative_sampling(targets=skip_targets,
                                                   max_range=max_entity_id,
                                                   num_sampled=num_false)

        return tf.cond(corrupt_head,
                       lambda: (corrupt_head, entity_table.lookup(tail),
                                relation_table.lookup(rel), sampled_true, sampled_false),
                       lambda: (corrupt_head, entity_table.lookup(head),
                                relation_table.lookup(rel), sampled_true, sampled_false),
                       name='corrupted_single_triple_w_multiple_targets')


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


def get_true_targets(entity, relation, targets_lookup_table, entity_table, targets, name=None):
    """

    :param entity:
    :param relation:
    :param targets_lookup_table:
    :param entity_table:
    :param targets:
    :param name:
    :return:
        A dense padded target matrix where the padded values are 0.
        A target_mask that has the same size of the target matrix with -1e10
            score if the [i,j] is not a padded score
    """
    with tf.name_scope(name, 'true_targets', [entity, relation, targets_lookup_table,
                                              entity_table, targets]):
        # Get a sparse tensor of target entities for each <entity, relation> pair
        targets = get_target_entities(entity=entity,
                                      relation=relation,
                                      targets_lookup_table=targets_lookup_table,
                                      entity_table=entity_table,
                                      targets=targets)

        targets_mask = tf.SparseTensor(targets.indices,
                                       tf.cast(tf.minimum(0, targets.values), tf.float32) * 1e10,
                                       targets.dense_shape)

        targets_dense = tf.sparse_tensor_to_dense(targets, default_value=0, name='true_targets_w_padding')

        return targets_dense, targets_mask


def get_true_tails(ent_rel_str, targets_lookup_table, targets, name=None):
    """
    Given ent \t rel pair return a list of string targets
    :param ent_rel_str:
    :param targets_lookup_table:
    :param name:
    :return:
    """
    with tf.name_scope(name, 'get_true_tails', [ent_rel_str, targets_lookup_table, targets]):
        target_entities_lookup_id = targets_lookup_table.lookup(ent_rel_str)
        # CHECK IF WE HAVE -1 HERE, if so the error will be have a -2 that is out of the range
        target_entities_lookup_id = tf.where(tf.equal(target_entities_lookup_id, -1),
                                             target_entities_lookup_id - 1,
                                             target_entities_lookup_id)
        # sparseTensor
        str_targets = tf.string_split(tf.nn.embedding_lookup(targets, target_entities_lookup_id), delimiter=' ')
        return str_targets.values
