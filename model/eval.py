import tensorflow as tf
import numpy as np
from model import data_helpers


# Files
tf.flags.DEFINE_string("test_premise_file", "", "test premise file")
tf.flags.DEFINE_string("test_hypothesis_file", "", "test hypothesis file")
tf.flags.DEFINE_string("test_label_file", "", "test label file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")

# Data Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("max_sequence_length", 100, "max sequence length")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab, idf = data_helpers.loadVocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))

SEQ_LEN = FLAGS.max_sequence_length
test_dataset = data_helpers.loadDataset(FLAGS.test_premise_file, FLAGS.test_hypothesis_file, FLAGS.test_label_file, vocab, SEQ_LEN)
print('test_dataset: {}'.format(len(test_dataset)))

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

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
        premise = graph.get_operation_by_name("premise").outputs[0]
        hypothesis   = graph.get_operation_by_name("hypothesis").outputs[0]

        premise_len = graph.get_operation_by_name("premise_len").outputs[0]
        hypothesis_len = graph.get_operation_by_name("hypothesis_len").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        model_extra_feature = graph.get_operation_by_name("extra_feature").outputs[0]

        premise_word_feature = graph.get_operation_by_name("premise_word_feature").outputs[0]
        hypothesis_word_feature   = graph.get_operation_by_name("hypothesis_word_feature").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

        num_test = 0
        prob_list = []
        target_list = [] 
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, idf, SEQ_LEN, shuffle=False)
        for test_batch in test_batches:
            x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, \
            targets, extra_feature, p_features, h_features = test_batch
            feed_dict = {
                premise: x_premise,
                hypothesis: x_hypothesis,
                premise_len: x_premise_len,
                hypothesis_len: x_hypothesis_len,
                dropout_keep_prob: 1.0,
                model_extra_feature: extra_feature,
                premise_word_feature: p_features,
                hypothesis_word_feature: h_features,
            }
            predicted_prob = sess.run(prob, feed_dict)
            prob_list.append(predicted_prob)
            target_list.append(targets)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))
            
probs_aggre = np.concatenate(prob_list, axis=0)
labels_aggre = np.concatenate(target_list, axis=0)  
        
prediction = np.argmax(probs_aggre, axis=1)
accuracy = np.equal(prediction, labels_aggre)
accuracy = np.mean(accuracy)

print('num_test_samples: {}  accuracy: {}'.format(num_test, round(accuracy, 3)))
