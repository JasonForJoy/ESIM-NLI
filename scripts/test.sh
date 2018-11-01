cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data

# latest_run=`ls -dt runs/* |head -n 1`
# latest_checkpoint=${latest_run}/checkpoints
latest_checkpoint=runs/1541064267/checkpoints
echo $latest_checkpoint

test_premise_file=$DATA_DIR/word_sequence/premise_snli_1.0_test.txt
test_hypothesis_file=$DATA_DIR/word_sequence/hypothesis_snli_1.0_test.txt
test_label_file=$DATA_DIR/word_sequence/label_snli_1.0_test.txt
vocab_file=$DATA_DIR/word_sequence/vocab.txt

batch_size=128
max_sequence_length=100

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python -u ${PKG_DIR}/model/eval.py \
				  --test_premise_file $test_premise_file \
                  --test_hypothesis_file $test_hypothesis_file \
                  --test_label_file $test_label_file \
                  --vocab_file $vocab_file \
                  --max_sequence_length $max_sequence_length \
                  --batch_size $batch_size \
                  --checkpoint_dir $latest_checkpoint # > log_test.txt 2>&1 &
