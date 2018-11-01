import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")
    # return tf.get_variable(initializer=initializer, name="word_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec

    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        else:
           embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 

    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

# output = tanh( xW + b )
def ffnn_layer(inputs, output_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope,  reuse=scope_reuse):
        input_size = inputs.get_shape()[-1].value
        W = tf.get_variable("W_trans", shape=[input_size, output_size], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b_trans", shape=[output_size, ], initializer=tf.zeros_initializer())
        outputs = tf.nn.relu(tf.einsum('aij,jk->aik', inputs, W) + b)
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
    return outputs

def premise_hypothesis_similarity_matrix(premise, hypothesis):
    #[batch_size, dim, p_len]
    p2 = tf.transpose(premise, perm=[0,2,1])

    #[batch_size, h_len, p_len]
    similarity = tf.matmul(hypothesis, p2, name='similarity_matrix')

    return similarity

def self_attended(similarity_matrix, inputs):
    #similarity_matrix: [batch_size, len, len]
    #inputs: [batch_size, len, dim]

    attended_w = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, len, dim]
    attended_out = tf.matmul(attended_w, inputs)
    return attended_out

def attend_hypothesis(similarity_matrix, premise, premise_len, maxlen):
    #similarity_matrix: [batch_size, h_len, p_len]
    #premise: [batch_size, p_len, dim]
    
    # masked similarity_matrix
    mask_p = tf.sequence_mask(premise_len, maxlen, dtype=tf.float32)   # [batch_size, p_len]
    mask_p = tf.expand_dims(mask_p, 1)                                 # [batch_size, 1, p_len]
    similarity_matrix = similarity_matrix * mask_p + -1e9 * (1-mask_p) # [batch_size, h_len, p_len]

    #[batch_size, h_len, p_len]
    attention_weight_for_p = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, a_len, dim]
    attended_hypothesis = tf.matmul(attention_weight_for_p, premise)
    return attended_hypothesis

def attend_premise(similarity_matrix, hypothesis, hypothesis_len, maxlen):
    #similarity_matrix: [batch_size, h_len, p_len]
    #hypothesis: [batch_size, h_len, dim]

    # masked similarity_matrix
    mask_h = tf.sequence_mask(hypothesis_len, maxlen, dtype=tf.float32)    # [batch_size, h_len]
    mask_h = tf.expand_dims(mask_h, 2)                                     # [batch_size, h_len, 1]
    similarity_matrix = similarity_matrix * mask_h + -1e9 * (1-mask_h)     # [batch_size, h_len, p_len]

    #[batch_size, p_len, h_len]
    attention_weight_for_h = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)

    #[batch_size, p_len, dim]
    attended_premise = tf.matmul(attention_weight_for_h, hypothesis)
    return attended_premise


class ESIM(object):
    def __init__(
      self, sequence_length, vocab_size, embedding_size, vocab, rnn_size, l2_reg_lambda=0.0):

        self.premise = tf.placeholder(tf.int32, [None, sequence_length], name="premise")
        self.hypothesis = tf.placeholder(tf.int32, [None, sequence_length], name="hypothesis")

        self.premise_len = tf.placeholder(tf.int32, [None], name="premise_len")
        self.hypothesis_len = tf.placeholder(tf.int32, [None], name="hypothesis_len")

        self.target = tf.placeholder(tf.int64, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.extra_feature = tf.placeholder(tf.float32, [None, 2], name="extra_feature")

        self.p_word_feature = tf.placeholder(tf.float32, [None, sequence_length, 2], name="premise_word_feature")
        self.h_word_feature = tf.placeholder(tf.float32, [None, sequence_length, 2], name="hypothesis_word_feature")

        l2_loss = tf.constant(0.0)

        # =============================== Embedding layer ===============================
        # 1. word embedding layer
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab) # tf.constant( np.array(vocab_size of task_dataset, dim) )
            premise_embedded = tf.nn.embedding_lookup(W, self.premise)  # [batch_size, q_len, word_dim]
            hypothesis_embedded = tf.nn.embedding_lookup(W, self.hypothesis)
        
        premise_embedded    = tf.nn.dropout(premise_embedded, keep_prob=self.dropout_keep_prob)
        hypothesis_embedded = tf.nn.dropout(hypothesis_embedded, keep_prob=self.dropout_keep_prob)
        print("shape of premise_embedded: {}".format(premise_embedded.get_shape()))
        print("shape of hypothesis_embedded: {}".format(hypothesis_embedded.get_shape()))

        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            p_rnn_output, p_rnn_states = lstm_layer(premise_embedded, self.premise_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)   # [batch_size, sequence_length, rnn_size(200)]
            premise_output = tf.concat(axis=2, values=p_rnn_output)     # [batch_size, maxlen, rnn_size*2]
            h_rnn_output, h_rnn_states = lstm_layer(hypothesis_embedded, self.hypothesis_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            hypothesis_output = tf.concat(axis=2, values=h_rnn_output)   # [batch_size, maxlen, rnn_size*2]
            print('Incorporate single_lstm_layer successfully.')
            
        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            similarity = premise_hypothesis_similarity_matrix(premise_output, hypothesis_output)  #[batch_size, answer_len, question_len]          
            attended_premise = attend_premise(similarity, hypothesis_output, self.hypothesis_len, sequence_length)  #[batch_size, maxlen, dim]
            attended_hypothesis = attend_hypothesis(similarity, premise_output, self.premise_len, sequence_length)  #[batch_size, maxlen, dim]

            m_p = tf.concat(axis=2, values=[premise_output, attended_premise, tf.multiply(premise_output, attended_premise), premise_output-attended_premise])
            m_h = tf.concat(axis=2, values=[hypothesis_output, attended_hypothesis, tf.multiply(hypothesis_output, attended_hypothesis), hypothesis_output-attended_hypothesis])
            
            # m_ffnn
            m_input_size = m_p.get_shape()[-1].value
            m_output_size = m_input_size
            m_p = ffnn_layer(m_p, m_output_size, self.dropout_keep_prob, "m_ffnn", scope_reuse=False)
            m_h = ffnn_layer(m_h, m_output_size, self.dropout_keep_prob, "m_ffnn", scope_reuse=True)
            print('Incorporate ffnn_layer after cross attention successfully.')

            rnn_scope_cross = 'bidirectional_rnn_cross'
            rnn_size_layer_2 = rnn_size
            rnn_output_p_2, rnn_states_p_2 = lstm_layer(m_p, self.premise_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=False)
            rnn_output_h_2, rnn_states_h_2 = lstm_layer(m_h, self.hypothesis_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)

            premise_output_cross    = tf.concat(axis=2, values=rnn_output_p_2)   # [batch_size, sequence_length, 2*rnn_size(400)]
            hypothesis_output_cross = tf.concat(axis=2, values=rnn_output_h_2)

        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            premise_max    = tf.reduce_max(premise_output_cross, axis=1)    # [batch_size, 2*rnn_size(400)]
            hypothesis_max = tf.reduce_max(hypothesis_output_cross, axis=1)

            premise_mean    = tf.reduce_mean(premise_output_cross, axis=1)    # [batch_size, 2*rnn_size(400)]
            hypothesis_mean = tf.reduce_mean(hypothesis_output_cross, axis=1)
        
            # premise_state    = tf.concat(axis=1, values=[rnn_states_p_2[0].h, rnn_states_p_2[1].h])   # [batch_size, 2*rnn_size(400)]
            # hypothesis_state = tf.concat(axis=1, values=[rnn_states_h_2[0].h, rnn_states_h_2[1].h])

            joined_feature =  tf.concat(axis=1, values=[premise_max, hypothesis_max, premise_mean, hypothesis_mean])  # [batch_size, 8*rnn_size(1600)]
            print("shape of joined feature: {}".format(joined_feature.get_shape()))

        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            #regularizer = None
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer")   # [batch_size, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)
            #full_out = tf.concat(axis=1, values=[full_out, self.extra_feature])
            
            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[3]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 3], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias   # [batch_size, 3]
            print("shape of logits: {}".format(logits.get_shape()))

            self.probs = tf.nn.softmax(logits, name="prob")   # [batch_size, n_class(3)]

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
