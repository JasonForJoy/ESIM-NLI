import os

base_dir = os.path.dirname(os.path.realpath(__file__))

vocab_file = os.path.join(base_dir, 'word_sequence/vocab.txt')
vocab = []
with open(vocab_file, 'rb') as f:
	for line in f:
		line = line.decode('utf-8').strip()
		vocab.append(line)     
print("Vocabulary size: {}".format(len(vocab)))


print("Filtering glove embedding ...")
glove_file = vocab_file = os.path.join(base_dir, 'glove/glove.840B.300d.txt')
vectors = {}
with open(glove_file, 'rt') as f:
    for line in f:
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, 300+1)]
        vectors[items[0]] = vec
print("Glove size: {}".format(len(vectors)))


filtered_vectors = {}
NOT = 0
for word in vocab:
	if word in vectors:
		filtered_vectors[word] = vectors[word]
	else:
		NOT += 1
print("Filtered vectors size: {}".format(len(filtered_vectors)))
print("Words not in glove size: {}".format(NOT))


filtered_glove_file = os.path.join(base_dir, 'glove/filtered_glove_840B_300d.txt')
with open(filtered_glove_file, 'w') as f:
	for word,vector in filtered_vectors.items():
		to_write = []
		to_write.append(word)
		vector = [str(ele) for ele in vector]
		to_write.extend(vector)
		f.write(" ".join(to_write))
		f.write("\n")
print("Write to {} finished.".format(filtered_glove_file))
