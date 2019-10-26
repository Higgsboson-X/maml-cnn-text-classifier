import numpy as np

import pickle
import os

from utils.vocab import Vocabulary

def meta_train_batch_generator(X_a, y_a, X_b, y_b, num_batches, support_data_size, query_data_size):

	# each input contains a list of inputs for all tasks with same data size
	# e.g., y_a = [y_a_t1, y_a_t2, ...]
	num_tasks = len(y_b)

	for i in range(num_tasks):
		assert y_a[i].shape[0] >= support_data_size and y_b[i].shape[0] >= query_data_size, "data not enough for task {}".format(i + 1)

	data_size_a = support_data_size
	data_size_b = query_data_size

	inds_a = np.array(list(range(data_size_a)))
	inds_b = np.array(list(range(data_size_b)))

	batch_size_a = data_size_a // num_batches
	batch_size_b = data_size_b // num_batches

	if data_size_a % num_batches:
		batch_size_a += 1
	if data_size_b % num_batches:
		batch_size_b += 1

	batch = 0

	while True:
		start = batch * batch_size_a
		end = min((batch + 1) * batch_size_a, data_size_a)
		X_batch_a = [X[inds_a[start:end]] for X in X_a]
		y_batch_a = [y[inds_a[start:end]] for y in y_a]

		start = batch * batch_size_b
		end = min((batch + 1) * batch_size_b, data_size_b)
		X_batch_b = [X[inds_b[start:end]] for X in X_b]
		y_batch_b = [y[inds_b[start:end]] for y in y_b]

		yield X_batch_a, y_batch_a, X_batch_b, y_batch_b

		batch += 1

		if batch == num_batches:
			batch = 0
			print("shuffling ...")
			np.random.shuffle(inds_a)
			np.random.shuffle(inds_b)


def fine_tune_batch_generator(X_train, y_train, num_batches):

	data_size = y_train.shape[0]
	inds = np.array(list(range(data_size)))
	batch_size = data_size // num_batches

	if data_size % num_batches:
		batch_size += 1

	batch = 0

	while True:
		start = batch * batch_size
		end = min((batch + 1) * batch_size, data_size)
		X_batch = X_train[inds[start:end]]
		y_batch = y_train[inds[start:end]]

		yield X_batch, y_batch

		batch += 1

		if batch == num_batches:
			batch = 0
			print("shuffling ...")
			np.random.shuffle(inds)


def get_text_sequences(text_file_path, vocab, max_seq_length, save_path=None):

	with open(text_file_path, 'r', encoding="utf-8") as f:
		lines = f.readlines()
		seqs = vocab.encode_sents(lines, length=max_seq_length)

	seqs = np.asarray(seqs, dtype="int32")
	print("encoded sequences shape: ", seqs.shape)

	if save_path is not None:
		with open(save_path, 'wb') as f:
			pickle.dump(seqs, f)

	return seqs


def get_text_label_pair(text_file_path, label_file_path, vocab, max_seq_length, save_path=None):

	with open(label_file_path, 'r', encoding="utf-8") as f:
		labels = []
		for line in f:
			labels.append(int(line.strip()))
		labels = np.asarray(labels, dtype="int32")

	seqs = get_text_sequences(text_file_path, vocab, max_seq_length)

	inds = list(range(labels.shape[0]))
	np.random.shuffle(inds)

	seqs = seqs[inds]
	labels = labels[inds]

	if save_path is not None:
		with open(save_path, 'wb') as f:
			pickle.dump({"seqs": seqs, "labels": labels}, f)

	return seqs, labels



def get_meta_train_data(mconf, vocab, num_tasks, save=False):

	X_a = []
	y_a = []
	X_b = []
	y_b = []

	for t in range(1, num_tasks + 1):
		
		print("loading data for task {} ...".format(t))

		train_text_file_path = mconf.data_dir_prefix + "t{}_text.train".format(t)
		train_label_file_path = mconf.data_dir_prefix + "t{}_label.train".format(t)
		seqs, labels = get_text_label_pair(train_text_file_path, train_label_file_path, vocab, mconf.max_seq_length)
		X_a.append(seqs)
		y_a.append(labels)

		print("support_data_size: ", seqs.shape, labels.shape)

		valid_text_file_path = mconf.data_dir_prefix + "t{}_text.val".format(t)
		valid_label_file_path = mconf.data_dir_prefix + "t{}_label.val".format(t)
		seqs, labels = get_text_label_pair(valid_text_file_path, valid_label_file_path, vocab, mconf.max_seq_length)
		X_b.append(seqs)
		y_b.append(labels)

		print("query_data_size: ", seqs.shape, labels.shape)

	if save:
		with open(mconf.processed_data_save_dir_prefix + "meta_a_{}t.pickle".format(num_tasks), 'wb') as f:
			pickle.dump({"X": X_a, "y": y_a}, f)
		with open(mconf.processed_data_save_dir_prefix + "meta_b_{}t.pickle".format(num_tasks), 'wb') as f:
			pickle.dump({"X": X_b, "y": y_b}, f)

		print("saved data to {}".format(mconf.processed_data_save_dir_prefix + "meta_train_x_{}t.pickle".format(num_tasks)))

	return X_a, y_a, X_b, y_b


def get_test_data(mconf, vocab, t):

	test_text_file_path = mconf.data_dir_prefix + "t{}_text.test".format(t)
	test_label_file_path = mconf.data_dir_prefix + "t{}_label.test".format(t)

	X, y = get_text_label_pair(test_text_file_path, test_label_file_path, vocab, mconf.max_seq_length)

	return X, y


def prepare_training_data_from_init(mconf, num_tasks=7):

	vocab_save_path = mconf.vocab_save_dir_prefix + "vocab.c{}".format(mconf.vocab_cutoff)
	vocab = Vocabulary()

	if os.path.exists(vocab_save_path):
		vocab.init_from_saved_vocab(vocab_save_path)
	else:
		vocab.update_vocab(mconf.data_dir_prefix + "all_text")
		if not os.path.exists(mconf.vocab_save_dir_prefix):
			os.makedirs(mconf.vocab_save_dir_prefix)
		vocab.save_vocab(vocab_save_path)

	mconf.vocab_size = vocab._size

	X_a, y_a, X_b, y_b = get_meta_train_data(mconf, vocab, num_tasks, save=True)

	return X_a, y_a, X_b, y_b, vocab



def prepare_training_data_from_file(mconf, num_tasks=7):

	vocab_save_path = mconf.vocab_save_dir_prefix + "vocab.c{}".format(mconf.vocab_cutoff)
	
	with open(vocab_save_path, "rb") as f:
		vocab = pickle.load(f)

	print("loading data from file ...")
	with open(mconf.processed_data_save_dir_prefix + "meta_a_{}t.pickle".format(num_tasks), 'rb') as f:
		d = pickle.load(f)
		X_a = d['X']
		y_a = d['y']
	with open(mconf.processed_data_save_dir_prefix + "meta_b_{}t.pickle".format(num_tasks), 'rb') as f:
		d = pickle.load(f)
		X_b = d['X']
		y_b = d['y']

	mconf.vocab_size = vocab._size
	print("vocab_size = {}".format(vocab._size))

	return X_a, y_a, X_b, y_b, vocab