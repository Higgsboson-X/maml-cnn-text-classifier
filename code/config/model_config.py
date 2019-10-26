class ModelConfig(object):

	def __init__(self):

		self.max_seq_length = 15
		self.vocab_size = 5000
		self.vocab_cutoff = 5
		self.filter_stopwords = False
		self.embedding_size = 300
		self.num_labels = 2

		self.support_data_size_per_task = 18000
		self.query_data_size_per_task = 1800

		self.num_filters = 128
		self.filter_sizes = [2, 3, 4, 5]

		self.l2_reg_weight = 0.03
		self.dropout_keep_prob = 0.8

		self.meta_lr = 0.001 # meta-learner
		self.train_lr = 0.01 # fine-tuning
		self.sub_lr = 0.01 # subtask

		self.corpus = "translations"

		self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
		self.vocab_save_dir_prefix = "../vocab/{}/".format(self.corpus)
		self.processed_data_save_dir_prefix = "../data/{}/processed/".format(self.corpus)
		self.data_dir_prefix = "../data/{}/tasks/".format(self.corpus)

	def init_from_dict(self, config):

		for key in config:
			setattr(self, key, config[key])

	def update_corpus(self):

		self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
		self.vocab_save_dir_prefix = "../vocab/{}/".format(self.corpus)
		self.processed_data_save_dir_prefix = "../data/{}/processed/".format(self.corpus)
		self.data_dir_prefix = "../data/{}/tasks/".format(self.corpus)


default_config = ModelConfig()