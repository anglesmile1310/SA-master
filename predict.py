#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

from tensorflow.python.eager import context
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import state_ops
# Parameters
# =====================================================================================================================

# Data Parameters       ===============================================

#Flag là cách truyền thông số vào chương trình để chạy model với nhiều cấu hình khác nhau.
tf.flags.DEFINE_string("positive_data_file", "./data1/Test/positive.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data1/Test/negative.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1530695379/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_path", "./runs/1530695379/checkpoints", "Checkpoint directory from training result")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
"""
def _remove_squeezable_dimensions(predictions, labels, weights):
  predictions = ops.convert_to_tensor(predictions)
  if labels is not None:
    labels, predictions = confusion_matrix.remove_squeezable_dimensions(
        labels, predictions)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is None:
    return predictions, labels, None

  weights = ops.convert_to_tensor(weights)
  weights_shape = weights.get_shape()
  weights_rank = weights_shape.ndims
  if weights_rank == 0:
    return predictions, labels, weights

  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  if (predictions_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - predictions_rank == 1:
      weights = array_ops.squeeze(weights, [-1])
    elif predictions_rank - weights_rank == 1:
      weights = array_ops.expand_dims(weights, [-1])
  else:
    # Use dynamic rank.
    weights_rank_tensor = array_ops.rank(weights)
    rank_diff = weights_rank_tensor - array_ops.rank(predictions)

    def _maybe_expand_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, -1),
          lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

    # Don't attempt squeeze if it will fail based on static check.
    if ((weights_rank is not None) and
        (not weights_shape.dims[-1].is_compatible_with(1))):
      maybe_squeeze_weights = lambda: weights
    else:
      maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

    def _maybe_adjust_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
          _maybe_expand_weights)

    # If weights are scalar, do nothing. Otherwise, try to add or remove a
    # dimension to match predictions.
    weights = control_flow_ops.cond(
        math_ops.equal(weights_rank_tensor, 0), lambda: weights,
        _maybe_adjust_weights)
  return predictions, labels, weights
def metric_variable(shape, dtype, validate_shape=True, name=None):

  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      trainable=False,
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      name=name)
def _count_condition(values,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None):

  check_ops.assert_type(values, dtypes.bool)
  count = metric_variable([], dtypes.float32, name='count')

  values = math_ops.to_float(values)
  if weights is not None:
    with ops.control_dependencies((check_ops.assert_rank_in(
        weights, (0, array_ops.rank(values))),)):
      weights = math_ops.to_float(weights)
      values = math_ops.multiply(values, weights)

  value_tensor = array_ops.identity(count)
  update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))

  if metrics_collections:
    ops.add_to_collections(metrics_collections, value_tensor)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return value_tensor, update_op

def false_positives(labels,
                    predictions,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.false_positives is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'false_positives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_false_positive = math_ops.logical_and(
        math_ops.equal(labels, False), math_ops.equal(predictions, True))
    return _count_condition(is_false_positive, weights, metrics_collections,
                            updates_collections)
def true_positives(labels,
                   predictions,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.true_positives is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'true_positives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_true_positive = math_ops.logical_and(
        math_ops.equal(labels, True), math_ops.equal(predictions, True))
    return _count_condition(is_true_positive, weights, metrics_collections,
                            updates_collections)

def precision(labels,
              predictions,
              weights=None,
              metrics_collections=None,
              updates_collections=None,
              name=None):
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.precision is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'precision',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    true_p, true_positives_update_op = true_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_p, false_positives_update_op = false_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_precision(tp, fp, name):
      return array_ops.where(
          math_ops.greater(tp + fp, 0), math_ops.div(tp, tp + fp), 0, name)

    p = compute_precision(true_p, false_p, 'value')
    update_op = compute_precision(true_positives_update_op,
                                  false_positives_update_op, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, p)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return p, update_op
"""

#FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
#checkpoint_dir="./runs/1530695379/"
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    print(x_raw)
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab.txt")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    y_test = np.argmax(y_test, axis=1)
   # print(y_test)
else:
    x_raw = ["tuyệt vời", "Giá hơi cao. Đành đợi 1 năm nữa rồi lấy em."]
    y_test = [1, 0]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab.txt")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#return đường dẫn(path) đầy đủ đến checjkpoint mới nhất
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    acc, acc_op = tf.metrics.accuracy(labels=y_test, predictions=all_predictions)

    rec1, rec_op1 = tf.metrics.precision(labels=y_test[0:int(len(y_test) / 2 + 1)], predictions=all_predictions[0:int(len(all_predictions) / 2 + 1)])
    rec0, rec_op0 = tf.metrics.precision(labels=y_test[int(len(y_test) / 2 + 1):], predictions=all_predictions[int(len(all_predictions) / 2 + 1):])
    pre1, pre_op1 = tf.metrics.recall(labels=y_test[0:int(len(y_test) / 2 + 1)],
                                         predictions=all_predictions[0:int(len(all_predictions) / 2 + 1)])
    pre0, pre_op0 = tf.metrics.recall(labels=y_test[int(len(y_test) / 2 + 1):],
                                         predictions=all_predictions[int(len(all_predictions) / 2 + 1):])


    # predict the class using your classifier
   # scorednn = list(DNNClassifier.predict_classes(input_fn=lambda: input_fn(testing_set)))
    #scoreArr = np.array(scorednn).astype(int)

    # run the session to compare the label with the prediction
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    v = sess.run(acc_op)  # accuracy

    r1 = sess.run(rec_op1)  # recall
    r0 = sess.run(rec_op0)  # recall
    p1 = sess.run(pre_op1)  # precision
    p0 = sess.run(pre_op0)  # precision

    print("accuracy ", v)
    """
    print("recall positive ", r1)
    print("recall nagetive ", r0)
    #print("precision positive ", p1)
    print("precision nagetive ", p0)
    print("F1 positive ", 2*p1*r1/(p1+r1))
    print("F1 nagetive ", 2*p0*r0/(p0+r0))
    """

    correct_predictions = float(sum(all_predictions == y_test))#y_test:nhãn đúng,all_predictions:nhãn dự đoán
    TP1=float(sum(all_predictions[:int(len(all_predictions)/2)+1]==y_test[:int(len(y_test)/2)+1]))
    TN1=float(sum(all_predictions[int(len(all_predictions)/2)+1:]==y_test[int(len(all_predictions)/2)+1:]))
    FN1=float(float(len(all_predictions)/2)-TP1)
    FP1=float(float(len(all_predictions)/2)-TN1)


    Precision_positive=TP1/(TP1+FP1)
    Recall_positive=TP1/(TP1+FN1)
    F1_positive=2*Precision_positive*Recall_positive/(Precision_positive+Recall_positive)
    Precision_nagetive = TN1 / (TN1 + FN1)
    Recall_nagetive = TN1 / (TN1 + FP1)
    F1_nagetive = 2 * Precision_nagetive * Recall_nagetive / (Precision_nagetive + Recall_nagetive)
    print("Precision positive: ",Precision_positive)
    print("Recall positive: ",Recall_positive)
    print("F1 positive: ",F1_positive)
    print("Precision nagetive: ", Precision_nagetive)
    print("Recall nagetive: ", Recall_nagetive)
    print("F1 nagetive: ", F1_nagetive)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w',encoding="UTF-8") as f:
    csv.writer(f).writerows(predictions_human_readable)
