import tensorflow as tf
import numpy as np
import datetime as dt

import os
import pickle

from config.model_config import default_config
from utils import data_processor

class MAMLCNNClassifier(object):

	def __init__(self, num_tasks=2, num_updates=2, mconf=default_config):

		self.mconf = mconf

		self.num_tasks = num_tasks
		self.num_updates = num_updates

		self.built = False


	def build_model(self):

		# support
		self.input_x_a = []
		self.input_y_a = []

		# query
		self.input_x_b = []
		self.input_y_b = []

		for i in range(1, self.num_tasks + 1):
			input_x_a = tf.placeholder(
				shape=[None, self.mconf.max_seq_length], dtype=tf.int32,
				name="input_x_a.{}".format(i)
			)
			input_y_a = tf.placeholder(
				shape=[None], dtype=tf.int32,
				name="input_y_a.{}".format(i)
			)

			input_x_b = tf.placeholder(
				shape=[None, self.mconf.max_seq_length], dtype=tf.int32,
				name="input_x_b.{}".format(i)
			)
			input_y_b = tf.placeholder(
				shape=[None], dtype=tf.int32,
				name="input_y_b.{}".format(i)
			)

			self.input_x_a.append(input_x_a)
			self.input_y_a.append(input_y_a)
			self.input_x_b.append(input_x_b)
			self.input_y_b.append(input_y_b)

		self.tmp_input_x = tf.placeholder(
			shape=[None, self.mconf.max_seq_length], dtype=tf.int32,
			name="tmp_input_x"
		)
		self.tmp_input_y = tf.placeholder(
			shape=[None], dtype=tf.int32,
			name="tmp_input_y"
		)

		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

		self.weights = self.construct_weights()

		inputs = [
			[
				self.input_x_a[i],
				self.input_y_a[i], 
				self.input_x_b[i], 
				self.input_y_b[i]
			] for i in range(self.num_tasks)
		]

		loss_a = []
		losses_b = []

		support_acc = []
		query_acc = []

		for i in range(self.num_tasks):
			print("building model for task {} ...".format(i + 1))
			# loss_a, losses_b, acc_a, query_acc
			res = self.meta_task(inputs[i])

			loss_a.append(res[0])
			losses_b.append(res[1])
			# query result after the first update
			support_acc.append(res[2])
			query_acc.append(res[3])

		losses_b = [[losses_b[i][j] for i in range(self.num_tasks)] for j in range(self.num_updates)]

		self.loss_a = sum(loss_a) / self.num_tasks
		self.losses_b = [sum(loss_b) / self.num_tasks for loss_b in losses_b]
		
		self.support_acc = sum(support_acc) / self.num_tasks
		self.query_acc = sum(query_acc) / self.num_tasks

		print("defining training operations ...")

		with tf.variable_scope("eval", reuse=tf.AUTO_REUSE):
			# accuracy for different tasks
			self.eval_ops = [self.support_acc, self.query_acc]

		with tf.variable_scope("meta_train", reuse=tf.AUTO_REUSE):
			optimizer = tf.train.AdamOptimizer(self.mconf.meta_lr)
			self.meta_train_ops = [
				optimizer.minimize(self.losses_b[-1]),
				self.loss_a,
				self.losses_b[-1],
				self.support_acc,
				self.query_acc
			]

		with tf.variable_scope("fine_tune", reuse=tf.AUTO_REUSE):
			optimizer = tf.train.AdamOptimizer(self.mconf.train_lr)
			pred_probs, pred_labels = self.forward_pass(self.tmp_input_x)
			loss = self.calc_loss(self.weights, pred_probs, self.tmp_input_y)
			accuracy = self.calc_acc(pred_labels, self.tmp_input_y)
			self.fine_tune_ops = [
				optimizer.minimize(loss),
				loss,
				accuracy
			]

		with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
			test_pred_probs, test_pred_labels = self.forward_pass(self.tmp_input_x)
			test_loss = self.calc_loss(self.weights, test_pred_probs, self.tmp_input_y)
			test_accuracy = self.calc_acc(test_pred_labels, self.tmp_input_y)
			self.test_ops = [test_loss, test_accuracy]

		with tf.variable_scope("infer", reuse=tf.AUTO_REUSE):
			infer_pred_probs, infer_pred_labels = self.forward_pass(self.tmp_input_x)
			self.infer_ops = [infer_pred_probs, infer_pred_labels]

		self.built = True

	def construct_weights(self):

		weights = dict()

		with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
			embedding_matrix = tf.get_variable(
				initializer=tf.random.uniform([self.mconf.vocab_size, self.mconf.embedding_size], -1.0, 1.0),
				trainable=True, name="embedding_matrix"
			)
			weights["embedding_matrix"] = embedding_matrix
		
		for i, filter_size in enumerate(self.mconf.filter_sizes):
			with tf.variable_scope("conv_maxpool_{}".format(filter_size), reuse=tf.AUTO_REUSE):
				filter_shape = [filter_size, self.mconf.embedding_size, 1, self.mconf.num_filters]
				kernel = tf.get_variable(
					initializer=tf.truncated_normal(filter_shape, stddev=0.1),
					name="kernel"
				)
				bias = tf.get_variable(
					initializer=tf.random_normal([self.mconf.num_filters], 0, 1),
					name="bias"
				)

				weights["conv_maxpool_{}_kernel".format(filter_size)] = kernel
				weights["conv_maxpool_{}_bias".format(filter_size)] = bias

		self.total_num_filters = self.mconf.num_filters * len(self.mconf.filter_sizes)
		with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
			kernel = tf.get_variable(
				initializer=tf.truncated_normal([self.total_num_filters, self.mconf.num_labels], stddev=0.1),
				name="kernel"
			)
			bias = tf.get_variable(
				initializer=tf.truncated_normal([self.mconf.num_labels], stddev=0.1),
				name="bias"
			)

			weights["output_kernel"] = kernel
			weights["output_bias"] = bias

		return weights


	def forward_pass(self, input_x, weights=None):

		if weights is None:
			weights = self.weights

		with tf.device("/cpu:0"), tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
			embedded_sequence = tf.nn.embedding_lookup(
				weights["embedding_matrix"], input_x
			)
			embedded_sequence_expanded = tf.expand_dims(embedded_sequence, -1)
		
		pooled_outputs = []
		for i, filter_size in enumerate(self.mconf.filter_sizes):
			with tf.variable_scope("conv_maxpool_{}".format(filter_size), reuse=tf.AUTO_REUSE):
				conv = tf.nn.conv2d(
					input=embedded_sequence_expanded,
					filter=weights["conv_maxpool_{}_kernel".format(filter_size)],
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv"
				)
				bias = weights["conv_maxpool_{}_bias".format(filter_size)]

				h = tf.nn.relu(tf.nn.bias_add(conv, bias))
				pool = tf.nn.avg_pool(
					value=h,
					ksize=[1, self.mconf.max_seq_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="pool"
				)
				pooled_outputs.append(pool)

		h_pool = tf.concat(pooled_outputs, 3)
		h_pool_flat = tf.reshape(h_pool, [-1, self.total_num_filters])

		with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
			h_dropout = tf.nn.dropout(
				h_pool_flat, self.dropout_keep_prob
			)
			pred_probs = tf.nn.softmax(
				tf.add(
					tf.matmul(h_dropout, weights["output_kernel"]), 
					weights["output_bias"]
				)
			)
			pred_labels = tf.argmax(pred_probs, axis=1, name="pred_labels")

		return pred_probs, pred_labels


	def meta_task(self, inputs):

		input_x_a, input_y_a, input_x_b, input_y_b = inputs

		pred_probs, pred_labels = self.forward_pass(input_x_a)

		loss_a = self.calc_loss(self.weights, pred_probs, input_y_a)
		support_acc = self.calc_acc(pred_labels, input_y_a)

		gvs = dict(zip(self.weights.keys(), tf.gradients(loss_a, list(self.weights.values()))))
		fast_weights = dict([(key, self.weights[key] - self.mconf.sub_lr * gvs[key]) for key in self.weights.keys() if key != "embedding_matrix"])
		fast_weights["embedding_matrix"] = tf.scatter_sub(
			ref=self.weights["embedding_matrix"],
			indices=gvs["embedding_matrix"].indices,
			updates=self.mconf.sub_lr * gvs["embedding_matrix"].values
		)

		losses_b = []

		pred_probs, pred_labels = self.forward_pass(input_x_b, weights=fast_weights)

		loss_b = self.calc_loss(fast_weights, pred_probs, input_y_b)
		losses_b.append(loss_b)

		# query after the first update
		query_acc = self.calc_acc(pred_labels, input_y_b)

		for _ in range(1, self.num_updates):
			pred_probs, _ = self.forward_pass(input_x_a, weights=fast_weights)
			loss = self.calc_loss(fast_weights, pred_probs, input_y_a)
			gvs = dict(zip(fast_weights.keys(), tf.gradients(loss, list(fast_weights.values()))))
			embedding_matrix = fast_weights["embedding_matrix"]
			fast_weights = dict([(key, fast_weights[key] - self.mconf.sub_lr * gvs[key]) for key in fast_weights.keys() if key != "embedding_matrix"])
			fast_weights["embedding_matrix"] = tf.scatter_sub(
				ref=embedding_matrix,
				indices=gvs["embedding_matrix"].indices,
				updates=self.mconf.sub_lr * gvs["embedding_matrix"].values
			)

			pred_probs, _ = self.forward_pass(input_x_b, weights=fast_weights)
			loss_b = self.calc_loss(fast_weights, pred_probs, input_y_b)
			losses_b.append(loss_b)

		return loss_a, losses_b, support_acc, query_acc


	def calc_loss(self, weights, pred_probs, true_labels):

		l2_loss = tf.nn.l2_loss(weights["output_kernel"]) + tf.nn.l2_loss(weights["output_bias"])
		loss = tf.reduce_mean(
			tf.keras.backend.categorical_crossentropy(
				target=tf.one_hot(true_labels, depth=self.mconf.num_labels, axis=-1),
				output=pred_probs
			)
		)

		return loss + self.mconf.l2_reg_weight * l2_loss


	def calc_acc(self, pred_labels, true_labels):

		correct_preds = tf.cast(tf.equal(tf.cast(pred_labels, dtype=tf.int32), true_labels), dtype=tf.float32)
		
		return tf.reduce_mean(correct_preds)


	def meta_train(self, sess, X_a, y_a, X_b, y_b, num_batches=64, epochs=10):

		batch_generator = data_processor.meta_train_batch_generator(
			X_a, y_a, X_b, y_b, num_batches, 
			self.mconf.support_data_size_per_task, self.mconf.query_data_size_per_task
		)

		for epoch in range(epochs):
			for i in range(num_batches):
				X_batch_a, y_batch_a, X_batch_b, y_batch_b = batch_generator.__next__()
				feed_dict = dict(
					list(zip(self.input_x_a, X_batch_a)) + \
					list(zip(self.input_y_a, y_batch_a)) + \
					list(zip(self.input_x_b, X_batch_b)) + \
					list(zip(self.input_y_b, y_batch_b))
				)
				feed_dict[self.dropout_keep_prob] = self.mconf.dropout_keep_prob

				[
					_,
					loss_a,
					last_loss_b,
					support_acc,
					query_acc
				] = sess.run(self.meta_train_ops, feed_dict=feed_dict)

				timestamp = dt.datetime.now().isoformat()
				print("{}: batch {}/{}, epoch {}/{}, loss_a {:g}, last_loss_b {:g}, support_acc {:g}, query_acc {:g}".format(
					timestamp, i + 1, num_batches, epoch + 1, epochs,
					loss_a, last_loss_b, support_acc, query_acc
				))

		print("completed meta training")

	def fine_tune(self, sess, X_train, y_train, X_valid, y_valid, num_batches=64, epochs=5, epochs_per_eval=1):

		batch_generator = data_processor.fine_tune_batch_generator(X_train, y_train, num_batches)

		for epoch in range(epochs):
			for i in range(num_batches):
				X_batch, y_batch = batch_generator.__next__()

				_, loss, accuracy = sess.run(self.fine_tune_ops, feed_dict={
					self.tmp_input_x: X_batch,
					self.tmp_input_y: y_batch,
					self.dropout_keep_prob: self.mconf.dropout_keep_prob
				})

				timestamp = dt.datetime.now().isoformat()
				print("{}: batch {}/{}, epoch {}/{}, loss {:g}, acc {:g}".format(
					timestamp, i + 1, num_batches, epoch + 1, epochs, loss, accuracy
				))

			if epoch % epochs_per_eval == 0:
				print("evaluations")
				print("--------------------")
				loss, accuracy = sess.run(self.test_ops, feed_dict={
					self.tmp_input_x: X_valid, self.tmp_input_y: y_valid, self.dropout_keep_prob: 1.0
				})
				print("loss = {:g}, accuracy = {:g}".format(loss, accuracy))


	def test(self, sess, X, y):

		test_loss, test_accuracy = sess.run(self.test_ops, feed_dict={
			self.tmp_input_x: X, self.tmp_input_y: y, self.dropout_keep_prob: 1.0
		})
		print("test")
		print("--------------------")
		print("loss = {:g}, accuracy = {:g}".format(test_loss, test_accuracy))

		return test_loss, test_accuracy


	def infer(self, sess, X):

		infer_pred_probs, infer_pred_labels = sess.run(self.infer_ops, feed_dict={self.tmp_input_x: X, self.dropout_keep_prob: 1.0})

		return infer_pred_probs, infer_pred_labels

	def save_model(self, sess, task_id="maml", timestamp=None):

		# task_id: "maml" / "t1", "t2", ...

		assert self.built, "model not built"

		saver = tf.train.Saver()
		if timestamp is None:
			timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
		model_save_dir = self.mconf.model_save_dir_prefix + timestamp
		if not os.path.exists(model_save_dir):
			os.makedirs(model_save_dir)
		saver.save(sess, model_save_dir + "/{}".format(task_id))
		print("saved model to {}".format(model_save_dir))

		return model_save_dir + "/{}".format(task_id), timestamp

	def restore_model(self, sess, path):

		assert self.built, "model not built"

		saver = tf.train.Saver()
		saver.restore(sess, path)

		print("model weights restored from {}".format(path))
