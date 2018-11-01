import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import data_helpers
from model.model_ESIM import ESIM
import operator
from collections import defaultdict

# Files
tf.flags.DEFINE_string("train_premise_file", "", "train premise file")
tf.flags.DEFINE_string("train_hypothesis_file", "", "train hypothesis file")
tf.flags.DEFINE_string("train_label_file", "", "train label file")
tf.flags.DEFINE_string("dev_premise_file", "", "dev premise file")
tf.flags.DEFINE_string("dev_hypothesis_file", "", "dev hypothesis file")
tf.flags.DEFINE_string("dev_label_file", "", "dev label file")
tf.flags.DEFINE_string("test_premise_file", "", "test premise file")
tf.flags.DEFINE_string("test_hypothesis_file", "", "test hypothesis file")
tf.flags.DEFINE_string("test_label_file", "", "test label file")
tf.flags.DEFINE_string("embedded_vector_file", "", "pre-trained embedded word vector")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000005, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_sequence_length", 200, "max sequence length")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
print("Loading data...")

# vocab = {"<PAD>": 0, ...}
vocab, idf = data_helpers.loadVocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))

SEQ_LEN = FLAGS.max_sequence_length
train_dataset = data_helpers.loadDataset(FLAGS.train_premise_file, FLAGS.train_hypothesis_file, FLAGS.train_label_file, vocab, SEQ_LEN)
print('train_dataset: {}'.format(len(train_dataset)))
dev_dataset = data_helpers.loadDataset(FLAGS.dev_premise_file, FLAGS.dev_hypothesis_file, FLAGS.dev_label_file, vocab, SEQ_LEN)
print('dev_dataset: {}'.format(len(dev_dataset)))
test_dataset = data_helpers.loadDataset(FLAGS.test_premise_file, FLAGS.test_hypothesis_file, FLAGS.test_label_file, vocab, SEQ_LEN)
print('test_dataset: {}'.format(len(test_dataset)))


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        esim = ESIM(
            sequence_length=SEQ_LEN,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            vocab=vocab,
            rnn_size=FLAGS.rnn_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                   5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(esim.mean_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        """
        loss_summary = tf.scalar_summary("loss", esim.mean_loss)
        acc_summary = tf.scalar_summary("accuracy", esim.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, 
                       targets, extra_feature, p_features, h_features):
            """
            A single training step
            """
            feed_dict = {
              esim.premise: x_premise,
              esim.hypothesis: x_hypothesis,
              esim.premise_len: x_premise_len,
              esim.hypothesis_len: x_hypothesis_len,
              esim.target: targets,
              esim.dropout_keep_prob: FLAGS.dropout_keep_prob,
              esim.extra_feature: extra_feature,
              esim.p_word_feature: p_features,
              esim.h_word_feature: h_features
            }

            _, step, loss, accuracy, predicted_prob = sess.run(
                [train_op, global_step, esim.mean_loss, esim.accuracy, esim.probs],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)


        def check_step(dataset, shuffle=False):
            results = defaultdict(list)
            num_test = 0
            num_correct = 0.0
            batches = data_helpers.batch_iter(dataset, FLAGS.batch_size, 1, idf, SEQ_LEN, shuffle=shuffle)
            for batch in batches:
                x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, \
                targets, extra_feature, p_features, h_features = batch
                feed_dict = {
                  esim.premise: x_premise,
                  esim.hypothesis: x_hypothesis,
                  esim.premise_len: x_premise_len,
                  esim.hypothesis_len: x_hypothesis_len,
                  esim.target: targets,
                  esim.dropout_keep_prob: 1.0,
                  esim.extra_feature: extra_feature,
                  esim.p_word_feature: p_features,
                  esim.h_word_feature: h_features
                }
                batch_accuracy, predicted_prob = sess.run([esim.accuracy, esim.probs], feed_dict)
                num_test += len(predicted_prob)
                if num_test % 1000 == 0:
                    print(num_test)

                num_correct += len(predicted_prob) * batch_accuracy

            # calculate Accuracy
            acc = num_correct / num_test
            print('num_test_samples: {}  accuracy: {}'.format(num_test, acc))

            return acc

        best_acc = 0.0
        EPOCH = 0
        batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, idf, SEQ_LEN, shuffle=True)
        for batch in batches:
            x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, \
            targets, extra_feature, p_features, h_features = batch
            train_step(x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, targets, extra_feature, p_features, h_features)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                EPOCH += 1
                print("\nEPOCH: {}".format(EPOCH))
                print("Evaluation on dev:")
                valid_acc = check_step(dev_dataset, shuffle=True)
                print("\nEvaluation on test:")
                test_acc = check_step(test_dataset, shuffle=False)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

