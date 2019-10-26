import argparse
import pprint

from config.model_config import ModelConfig

def load_args():

	parser = argparse.ArgumentParser(
		prog="MAML_CNN_CLS", 
		description="MAML-CNN-Text-Classifier"
	)
	# required for mconf
	parser.add_argument(
		"--max-seq-length", type=int, default=15, 
		help="maximum sequence length for sentences"
	)
	parser.add_argument(
		"--vocab-cutoff", type=int, default=5, 
		help="maximum word count to ignore in vocabulary"
	)
	parser.add_argument(
		"--filter-stopwords", action="store_true", help="whether to filter stopwords"
	)
	parser.add_argument(
		"--embedding-size", type=int, default=300, 
		help="dimension of sentence embedding"
	)
	parser.add_argument(
		"--num-labels", type=int, default=2,
		help="total number of labels per task"
	)
	parser.add_argument(
		"--support-data-size-per-task", type=int, default=18000,
		help="support data size for each task"
	)
	parser.add_argument(
		"--query-data-size-per-task", type=int, default=1800,
		help="query data size for each task"
	)
	parser.add_argument(
		"--num-filters", type=int, default=128,
		help="number of filters for each size"
	)
	parser.add_argument(
		"--l2-reg-weight", type=float, default=0.03,
		help="l2 regularization weight for output kernel and bias"
	)
	parser.add_argument(
		"--dropout-keep-prob", type=float, default=0.8,
		help="keep probability of dropout layer, only for training"
	)
	parser.add_argument(
		"--meta-lr", type=float, default=0.001,
		help="learning rate for meta learner in meta-training"
	)
	parser.add_argument(
		"--train-lr", type=float, default=0.01,
		help="learning rate for fine-tuning"
	)
	parser.add_argument(
		"--sub-lr", type=float, default=0.01,
		help="learning rate for sub-tasks in meta-training"
	)
	parser.add_argument(
		"--corpus", type=str, default="translations",
		help="name of corpus, for defining directions and paths"
	)

	# global options
	parser.add_argument(
		"--maml-epochs", type=int, default=10,
		help="meta-training epochs, 0 if normal training scheme"
	)
	parser.add_argument(
		"--train-epochs", type=int, default=0,
		help="fine-tuning epochs"
	)
	parser.add_argument(
		"--maml-num-batches", type=int, default=64,
		help="number of batches for maml-training"
	)
	parser.add_argument(
		"--train-num-batches", type=int, default=64,
		help="number of batches for fine-tuning"
	)
	parser.add_argument(
		"--num-tasks", type=int, default=7,
		help="total number of sub-tasks"
	)
	parser.add_argument(
		"--num-updates", type=int, default=2,
		help="number of updates for each sub-task"
	)
	parser.add_argument(
		"--train-task-id", type=int, default=0,
		help="task id for fine-tuning, 0 for all tasks available"
	)
	parser.add_argument(
		"--test-task-id", type=int, default=-1,
		help="task id for test after meta-training, -1 for none, 0 for all"
	)
	parser.add_argument(
		"--infer-task-id", type=int, default=-1,
		help="task id for inference after meta-training, -1 for none, 0 for all"
	)
	parser.add_argument(
		"--epochs-per-eval", type=int, default=1,
		help="number of epochs per evaluation in fine-tuning"
	)
	parser.add_argument(
		"--infer", action="store_true", help="whether to perform an inference after fine-tuning"
	)
	parser.add_argument(
		"--test", action="store_true", help="whether to perform a test after fine-tuning"
	)
	parser.add_argument(
		"--restore-ckpt-path", type=str, default='',
		help="checkpoint path for saved model if restore from saved"
	)

	args = parser.parse_args()

	return args


def build_mconf_from_args(args):

	mconf = ModelConfig()

	mconf = ModelConfig()
	for attr in vars(mconf).keys():
		if hasattr(args, attr):
			setattr(mconf, attr, getattr(args, attr))

	mconf.update_corpus()

	return mconf


if __name__ == "__main__":

	args = load_args()
	mconf = build_mconf_from_args(args)

	printer = pprint.PrettyPrinter(indent=4)
	printer.pprint(vars(args))
	printer.pprint(vars(mconf))

	
