import tensorflow as tf
import numpy as np
import datetime as dt

import os
import pickle

from config.model_config import default_config

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

		for i in range(1, self.mconf.num_tasks + 1):
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

		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

		self.weights = self.construct_weights()

		inputs_a = [[self.input_x_a[i], self.input_y_a[i]] for i in range(self.num_tasks)]
		inputs_b = [[self.input_x_b[i], self.input_y_b[i]] for i in range(self.num_tasks)]

		loss_a = []
		losses_b = []
		results_b = []

		for i in range(self.num_tasks):
			print("building model for task {} ...".format(i + 1))
			inputs = [inputs_a[i], inputs_b[i]]
			results = self.meta_task(inputs)

			loss_a.append(results[0])
			losses_b.append(results[1])
			results_b.append(results[2])

		losses_b = [[losses_b[i][j] for i in range(self.num_tasks)] for j in range(self.num_updates)]

		self.loss_a = sum(loss_a) / self.num_tasks
		self.losses_b = [sum(loss_b) / self.num_tasks for loss_b in losses_b]

		print("defining training operations ...")

		with tf.variable_scope("eval", reuse=tf.AUTO_REUSE):
			# accuracy for different tasks
			

		with tf.variable_scope("train", reuse=tf.AUTO_REUSE):


		with tf.variable_scope("test", reuse=tf.AUTO_REUSE):


		with tf.variable_scope("infer", reuse=tf.AUTO_REUSE):


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
				initializer=tf.truncated_normal([total_num_filters, self.mconf.num_labels], stddev=0.1),
				name="kernel"
			)
			bias = tf.get_variable(
				initializer=tf.truncated_normal([self.mconf.num_labels], stddev=0.1),
				name="bias"
			)

			weights["output_kernel"] = kernel
			weights["output_bias"] = bias

		return weights


	def forward_pass(self, inputs, weights=None):

		ipnut_x, input_y = inputs
		ipnut_y_onehot = tf.one_hot(input_y, depth=self.mconf.num_labels, axis=-1)

		if weights is None:
			weights = self.weights

		with tf.device("/cpu:0"), tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
			embedded_sequence = tf.nn.embedding_lookup(
				weights["embedding_matrix"], inputs_x
			)
			embedded_sequence_expanded = tf.expand_dims(embedding_sequence, -1)
		
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

		l2_loss = tf.nn.l2_loss(weights["output_kernel"]) + tf.nn.l2_loss(weights["output_bias"])

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


		losses = tf.keras.backend.categorical_crossentropy(
			target=ipnut_y_onehot, output=pred_probs
		)
		loss_all = tf.reduce_mean(losses) + l2_reg_weight * l2_loss
		correct_preds = tf.cast(tf.equal(tf.cast(pred_labels, dtype=tf.int32), input_y), dtype=tf.float32)
		accuracy = tf.reduce_mean(correct_preds)

		return [pred_probs, pred_labels, accuracy], loss_all


	def meta_task(self, inputs):

		inputs_a, inputs_b = inputs

		_, loss_a = self.forward_pass(inputs_a)

		gvs = dict(zip(self.weights.keys(), tf.gradients(loss_a, list(self.weights.values()))))
		fast_weights = dict([(key, self.weights[key] - self.mconf.sub_lr * gvs[key]) for key in self.weights.keys()])

		losses_b = []
		results_b = []

		result_b, loss_b = self.forward_pass(inputs_b, weights=fast_weights)

		losses_b.append(loss_b)
		results_b.append(result_b)

		for _ in range(1, self.num_updates):
			_, loss = self.forward_pass(inputs_a)
			gvs = dict(zip(self.weights.keys(), tf.gradients(loss, list(self.weights.values()))))
			fast_weights = dict([(key, fast_weights[key] - self.mconf.sub_lr * gvs[key]) for key in fast_weights.keys()])

			result_b, loss_b = self.forward_pass(inputs_b, weights=fast_weights)

			losses_b.append(loss_b)
			results_b.append(result_b)

		return loss_a, losses_b, results_b
