import os
import cPickle

base_dir = os.path.dirname(os.path.realpath(__file__))
dictionary = os.path.join(base_dir, 'word_sequence/vocab_cased.pkl')
vocab = os.path.join(base_dir, 'word_sequence/vocab.txt')

with open(dictionary, 'rb') as f:
	worddicts = cPickle.load(f)

with open(vocab, 'w') as f:
	for k, v in worddicts.items():
		f.write(k)
		f.write('\n')
	print("Preprocess vocab done.")
