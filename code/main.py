import imp
import pickle
import pprint
import tensorflow as tf
import numpy as np

import arguments
import models.maml_cnn_classifier
import config.model_config
import utils.data_processor
import utils.vocab

def run():

	args = arguments.load_args()
	mconf = arguments.build_mconf_from_args(args)

	printer = pprint.PrettyPrinter(indent=4)

	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	print("loading data and vocabulary ...")
	try:
		X_a, y_a, X_b, y_b, vocab = utils.data_processor.prepare_training_data_from_file(mconf, num_tasks=args.num_tasks)
	except:
		X_a, y_a, X_b, y_b, vocab = utils.data_processor.prepare_training_data_from_init(mconf, num_tasks=args.num_tasks)

	print(">>>>>>> Model Config <<<<<<<")
	printer.pprint(vars(mconf))

	model = models.maml_cnn_classifier.MAMLCNNClassifier(num_tasks=args.num_tasks, num_updates=args.num_updates, mconf=mconf)
	model.build_model()
	sess = tf.Session()
	if args.restore_ckpt_path != '':
		try:
			model.restore_model(sess, args.restore_ckpt_path)
		except:
			raise Exception("failed to load model")
	else:
		sess.run(tf.global_variables_initializer())

	timestamp = None

	# meta-learning
	if args.maml_epochs > 0:
		assert len(X_a) == len(y_a) == len(X_b) == len(y_b) == args.num_tasks, "data size not satisfied for {} tasks".format(args.num_tasks)
		print("training meta-learner ...")
		model.meta_train(sess, X_a, y_a, X_b, y_b, 
			num_batches=args.maml_num_batches, epochs=args.maml_epochs
		)
		init_model_path, timestamp = model.save_model(sess, task_id="maml", timestamp=None)

	if args.test_task_id >= 0:
		if args.test_task_id == 0:
			to_test = list(range(1, args.num_tasks + 1))
		else:
			assert 1 <= args.test_task_id <= args.num_tasks, "task_id out of range"
			to_test = [args.test_task_id]
		test_results = dict()
		for task_id in to_test:
			X_test, y_test = utils.data_processor.get_test_data(mconf, vocab, task_id)
			test_loss, test_accuracy = model.test(sess, X_test, y_test)
			test_results["t{}".format(task_id)] = {
				"loss": test_loss,
				"accuracy": test_accuracy
			}
		with open("results_maml.test", 'wb') as f:
			pickle.dump(test_results, f)
			print("saved test results to results.test")
	if args.infer_task_id >= 0:
		if args.infer_task_id == 0:
			to_infer = list(range(1, args.num_tasks + 1))
		else:
			assert 1 <= args.infer_task_id <= args.num_tasks, "task_id out of range"
			to_infer = [args.infer_task_id]
		for task_id in to_infer:
			infer_text_file_path = mconf.data_dir_prefix + "t{}_text.infer".format(task_id)
			X_infer = utils.data_processor.get_text_sequences(infer_text_file_path, vocab, mconf.max_seq_length)
			infer_pred_probs, infer_pred_labels = model.infer(sess, X_infer)
			with open("t{}_output_maml.infer".format(task_id), 'wb') as f:
				pickle.dump({"probs": infer_pred_probs, "labels": infer_pred_labels}, f)
				print("saved output to t{}_output_maml.infer".format(task_id))

	# fine-tuning
	if args.train_epochs > 0:
		if args.train_task_id == 0:
			to_train = list(range(1, args.num_tasks + 1))
		else:
			assert 1 <= args.train_task_id <= args.num_tasks, "task_id out of range"
			to_train = [args.train_task_id]
		for task_id in to_train:
			print("training task {} ...".format(task_id))
			model.fine_tune(sess, X_a[task_id-1], y_a[task_id-1], X_b[task_id-1], y_b[task_id-1], 
				num_batches=args.train_num_batches, epochs=args.train_epochs, 
				epochs_per_eval=args.epochs_per_eval
			)
			_, timestamp = model.save_model(sess, task_id="t{}".format(task_id), timestamp=timestamp)

			if args.test:
				try:
					X_test, y_test = utils.data_processor.get_test_data(mconf, vocab, task_id)
					test_loss, test_accuracy = model.test(sess, X_test, y_test)
				except:
					print("error when testing task {}, ignoring ...".format(task_id))
			if args.infer:
				try:
					infer_text_file_path = mconf.data_dir_prefix + "t{}_text.infer".format(task_id)
					X_infer = utils.data_processor.get_text_sequences(infer_text_file_path, vocab, mconf.max_seq_length)
					infer_pred_probs, infer_pred_labels = model.infer(sess, X_infer)
					with open("t{}_output_train.infer".format(task_id), 'wb') as f:
						pickle.dump({"probs": infer_pred_probs, "labels": infer_pred_labels}, f)
						print("saved output to t{}_output_train.infer".format(task_id))
				except:
					print("error when inferring task {}, ignoring ...".format(task_id))

			model.restore_model(sess, init_model_path)

	

	print(">>>>>>> Completed <<<<<<<")


if __name__ == "__main__":

	run()