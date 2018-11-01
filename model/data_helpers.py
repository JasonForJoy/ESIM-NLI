import numpy as np
import random


def loadVocab(fname):
    '''
    vocab = {"<PAD>": 0, ...}
    idf   = { 0: log(total_doc/doc_freq)}
    '''
    vocab={}
    idf={}
    with open(fname, 'rt') as f:
        for index, word in enumerate(f):
            word = word.decode('utf-8').strip()
            vocab[word] = index
    return vocab, idf

def toVec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["_UNK_"])

    return length, np.array(vec)


def loadDataset(premise_file, hypothesis_file, label_file, vocab, maxlen):

    # premise
    premise_tokens = []
    premise_vec = []
    premise_len = []
    with open(premise_file, 'rt') as f1:
        for line in f1:
            line = line.decode('utf-8').strip()
            p_tokens = line.split(' ')[:maxlen]
            p_len, p_vec = toVec(p_tokens, vocab, maxlen)
            premise_tokens.append(p_tokens)
            premise_vec.append(p_vec)
            premise_len.append(p_len)

    # hypothesis
    hypothesis_tokens = []
    hypothesis_vec = []
    hypothesis_len = []
    with open(hypothesis_file, 'rt') as f2:
        for line in f2:
            line = line.decode('utf-8').strip()
            h_tokens = line.split(' ')[:maxlen]
            h_len, h_vec = toVec(h_tokens, vocab, maxlen)
            hypothesis_tokens.append(h_tokens)
            hypothesis_vec.append(h_vec)
            hypothesis_len.append(h_len)

    # label
    label = []
    with open(label_file, 'rt') as f3:
        for line in f3:
            line = line.decode('utf-8').strip()
            label.append(int(line))

    assert len(premise_tokens) == len(hypothesis_tokens)
    assert len(hypothesis_tokens) == len(label)

    # dataset
    dataset = []
    for i in range(len(label)):
        dataset.append( (premise_tokens[i], premise_vec[i], premise_len[i],
                         label[i], 
                         hypothesis_tokens[i], hypothesis_vec[i], hypothesis_len[i]) )

    return dataset


def word_count(q_vec, a_vec, q_len, a_len, idf):
    q_set = set([q_vec[i] for i in range(q_len) if q_vec[i] > 100])
    a_set = set([a_vec[i] for i in range(a_len) if a_vec[i] > 100])
    new_q_len = float(max(len(q_set), 1))
    count1 = 0.0
    count2 = 0.0
    for id1 in q_set:
        if id1 in a_set:
            count1 += 1.0
            if id1 in idf:
                count2 += idf[id1]
    return count1/new_q_len, count2/new_q_len

def common_words(q_vec, a_vec, q_len, a_len):
    q_set = set([q_vec[i] for i in range(q_len) if q_vec[i] > 100])
    a_set = set([a_vec[i] for i in range(a_len) if a_vec[i] > 100])
    return q_set.intersection(a_set)

def tfidf_feature(id_list, common_id_set, idf):
    word_freq={}
    for t in id_list:
        if t in common_id_set:
            if t in word_freq:
                word_freq[t] += 1
            else:
                word_freq[t] = 1
    tfidf_feature={}
    for t in common_id_set:
        if t in idf:
            tfidf_feature[t] = word_freq[t] * idf[t]
        else:
            tfidf_feature[t] = word_freq[t]
    return tfidf_feature

def word_feature(id_list, tfidf):
    len1 = len(id_list)
    features = np.zeros((len1, 2), dtype='float32')
    for idx, t in enumerate(id_list):
        if t in tfidf:
            features[idx, 0] = 1
            features[idx, 1] = tfidf[t]
    return features

def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def batch_iter(data, batch_size, num_epochs, idf, maxlen, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_premise = []
            x_hypothesis = []
            x_premise_len = []
            x_hypothesis_len = []

            targets = []
            p_features=[]
            h_features=[]
            extra_feature =[]

            for rowIdx in range(start_index, end_index):
                premise_tokens, premise_vec, premise_len,\
                label, \
                hypothesis_tokens, hypothesis_vec, hypothesis_len = data[rowIdx]

                # feature 1
                word_count_feature1, word_count_feature2 = word_count(premise_vec, hypothesis_vec, premise_len, hypothesis_len, idf)  # scalar feature
                common_ids = common_words(premise_vec, hypothesis_vec, premise_len, hypothesis_len)    # list: q_set.intersection(a_set) when word_id > 100
                tfidf = tfidf_feature(premise_vec, common_ids, idf)            # dict: { id: scalar feature }

                # normalize premise_vec and hypothesis_vec
                new_premise_vec = normalize_vec(premise_vec, maxlen)    # pad the original vec to the same maxlen
                new_hypothesis_vec = normalize_vec(hypothesis_vec, maxlen)

                # feature 2
                p_word_feature = word_feature(new_premise_vec, tfidf)   # feature of np.array( maxlen, 2 ) 
                h_word_feature = word_feature(new_hypothesis_vec, tfidf)

                x_premise.append(new_premise_vec)
                x_premise_len.append(premise_len)
                x_hypothesis.append(new_hypothesis_vec)
                x_hypothesis_len.append(hypothesis_len)
                targets.append(label)

                p_features.append(p_word_feature)
                h_features.append(h_word_feature)

                extra_feature.append(np.array([word_count_feature1, word_count_feature2], dtype="float32") )

            yield np.array(x_premise), np.array(x_hypothesis), np.array(x_premise_len), np.array(x_hypothesis_len),\
                  np.array(targets), np.array(extra_feature), np.array(p_features), np.array(h_features)

