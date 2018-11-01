cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data

train_premise_file=$DATA_DIR/word_sequence/premise_snli_1.0_train.txt
train_hypothesis_file=$DATA_DIR/word_sequence/hypothesis_snli_1.0_train.txt
train_label_file=$DATA_DIR/word_sequence/label_snli_1.0_train.txt

dev_premise_file=$DATA_DIR/word_sequence/premise_snli_1.0_dev.txt
dev_hypothesis_file=$DATA_DIR/word_sequence/hypothesis_snli_1.0_dev.txt
dev_label_file=$DATA_DIR/word_sequence/label_snli_1.0_dev.txt

test_premise_file=$DATA_DIR/word_sequence/premise_snli_1.0_test.txt
test_hypothesis_file=$DATA_DIR/word_sequence/hypothesis_snli_1.0_test.txt
test_label_file=$DATA_DIR/word_sequence/label_snli_1.0_test.txt

embedded_vector_file=$DATA_DIR/glove/filtered_glove_840B_300d.txt
vocab_file=$DATA_DIR/word_sequence/vocab.txt

lambda=0
dropout_keep_prob=0.8
batch_size=128
max_sequence_length=100
DIM=300
rnn_size=300
evaluate_every=4292

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python -u ${PKG_DIR}/model/train.py \
                --train_premise_file $train_premise_file \
                --train_hypothesis_file $train_hypothesis_file \
                --train_label_file $train_label_file \
                --dev_premise_file $dev_premise_file \
                --dev_hypothesis_file $dev_hypothesis_file \
                --dev_label_file $dev_label_file \
                --test_premise_file $test_premise_file \
                --test_hypothesis_file $test_hypothesis_file \
                --test_label_file $test_label_file \
                --embedded_vector_file $embedded_vector_file \
                --vocab_file $vocab_file \
                --max_sequence_length $max_sequence_length \
                --embedding_dim $DIM \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --batch_size $batch_size \
                --rnn_size $rnn_size \
                --evaluate_every $evaluate_every # > log.txt 2>&1 &
