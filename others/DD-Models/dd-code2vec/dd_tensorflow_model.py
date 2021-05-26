import tensorflow as tf
from typing import Dict, Optional

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors, EstimatorAction
from common import common
from vocabularies import VocabType
from config import Config
from dd_model_base import Code2VecModelBase

tf.compat.v1.disable_eager_execution()


class Code2VecModel(Code2VecModelBase):
    def __init__(self, config: Config):
        self.sess = tf.compat.v1.Session()
        self.saver = None

        self.eval_reader = None
        self.eval_input_iterator_reset_op = None
        self.predict_reader = None

        # self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_code_vectors, \
            self.eval_attentions_op, self.eval_losses_op = None, None, None, None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None

        self.vocab_type_to_tf_variable_name_mapping: Dict[VocabType, str] = {
            VocabType.Token: 'WORDS_VOCAB',
            VocabType.Target: 'TARGET_WORDS_VOCAB',
            VocabType.Path: 'PATHS_VOCAB'
        }

        super(Code2VecModel, self).__init__(config)

        # move from evaluate to here
        if self.eval_reader is None:
            self.eval_reader = PathContextReader(vocabs=self.vocabs,
                                                 model_input_tensors_former=_TFEvaluateModelInputTensorsFormer(),
                                                 config=self.config, estimator_action=EstimatorAction.Evaluate)
            input_iterator = tf.compat.v1.data.make_initializable_iterator(self.eval_reader.get_dataset())
            self.eval_input_iterator_reset_op = input_iterator.initializer
            input_tensors = input_iterator.get_next()

            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, _, _, \
                self.eval_code_vectors, self.eval_attentions_op, self.eval_losses_op = \
                self._build_tf_test_graph(input_tensors, normalize_scores=True)
            if self.saver is None:
                self.saver = tf.compat.v1.train.Saver()

        if self.config.MODEL_LOAD_PATH and not self.config.TRAIN_DATA_PATH_PREFIX:
            self._initialize_session_variables()
            self._load_inner_model(self.sess)

    def evaluate(self) -> Optional[str]:
        self.sess.run(self.eval_input_iterator_reset_op)
        # Run evaluation in a loop until iterator is exhausted.
        # Each iteration = batch. We iterate as long as the tf iterator (reader) yields batches.
        try:
            while True:
                top_words, top_scores, original_names, code_vectors, losses = self.sess.run(
                    [self.eval_top_words_op, self.eval_top_values_op,
                     self.eval_original_names_op, self.eval_code_vectors, self.eval_losses_op],
                )

                # shapes:
                #   top_words: (batch, top_k);   top_scores: (batch, top_k)
                #   original_names: (batch, );   code_vectors: (batch, code_vector_size)

                top_words = common.binary_to_string_matrix(top_words)  # (batch, top_k)
                original_names = common.binary_to_string_list(original_names)  # (batch,)

                return self._log_predictions_during_evaluation(zip(original_names, top_words, top_scores, losses))
        except tf.errors.OutOfRangeError:
            pass  # reader iterator is exhausted and have no more batches to produce.

    def get_attention(self) -> Optional[str]:
        self.sess.run(self.eval_input_iterator_reset_op)
        # Run evaluation in a loop until iterator is exhausted.
        try:
            while True:
                attentions, losses = self.sess.run(
                    [self.eval_attentions_op, self.eval_losses_op],
                )
                for sub_attr in attentions:
                    return sub_attr  # only first sub token
        except tf.errors.OutOfRangeError:
            pass  # reader iterator is exhausted and have no more batches to produce.

    def _calculate_weighted_contexts(self, tokens_vocab, paths_vocab, attention_param, source_input, path_input,
                                     target_input, valid_mask, is_evaluating=False):
        source_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=source_input)  # (batch, max_contexts, dim)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=path_input)  # (batch, max_contexts, dim)
        target_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=target_input)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
                                  axis=-1)  # (batch, max_contexts, dim * 3)

        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, rate=1 - self.config.DROPOUT_KEEP_RATE)

        flat_embed = tf.reshape(context_embed, [-1, self.config.context_vector_size])  # (batch * max_contexts, dim * 3)
        transform_param = tf.compat.v1.get_variable(
            'TRANSFORM', shape=(self.config.context_vector_size, self.config.CODE_VECTOR_SIZE), dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

        contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
        batched_contexts_weights = tf.reshape(
            contexts_weights, [-1, self.config.MAX_CONTEXTS, 1])  # (batch, max_contexts, 1)
        mask = tf.math.log(valid_mask)  # (batch, max_contexts)
        mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
        batched_contexts_weights += mask  # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)  # (batch, max_contexts, 1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, self.config.MAX_CONTEXTS, self.config.CODE_VECTOR_SIZE])
        code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)  # (batch, dim * 3)

        return code_vectors, attention_weights

    def _build_tf_test_graph(self, input_tensors, normalize_scores=False):
        with tf.compat.v1.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Token],
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            targets_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Target],
                shape=(self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            attention_param = tf.compat.v1.get_variable(
                'ATTENTION', shape=(self.config.context_vector_size, 1),
                dtype=tf.float32, trainable=False)
            paths_vocab = tf.compat.v1.get_variable(
                self.vocab_type_to_tf_variable_name_mapping[VocabType.Path],
                shape=(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)

            targets_vocab = tf.transpose(targets_vocab)  # (dim * 3, target_word_vocab)

            # Use `_TFEvaluateModelInputTensorsFormer` to access input tensors by name.
            input_tensors = _TFEvaluateModelInputTensorsFormer().from_model_input_form(input_tensors)
            # shape of (batch, 1) for input_tensors.target_string
            # shape of (batch, max_contexts) for the other tensors

            code_vectors, attention_weights = self._calculate_weighted_contexts(
                tokens_vocab, paths_vocab, attention_param, input_tensors.path_source_token_indices,
                input_tensors.path_indices, input_tensors.path_target_token_indices,
                input_tensors.context_valid_mask, is_evaluating=True)

        scores = tf.matmul(code_vectors, targets_vocab)  # (batch, target_word_vocab)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(input_tensors.target_index, [-1]),
            logits=scores)

        topk_candidates = tf.nn.top_k(scores, k=tf.minimum(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION, self.vocabs.target_vocab.size))
        top_indices = topk_candidates.indices
        top_words = self.vocabs.target_vocab.lookup_word(top_indices)
        original_words = input_tensors.target_string
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, input_tensors.path_source_token_strings, \
            input_tensors.path_strings, input_tensors.path_target_token_strings, \
               code_vectors, attention_weights, losses

    def _load_inner_model(self, sess=None):
        if sess is not None:
            self.log('Loading model weights from: ' + self.config.MODEL_LOAD_PATH)
            self.saver.restore(sess, self.config.MODEL_LOAD_PATH)
            self.log('Done loading model weights')

    def get_should_reuse_variables(self):
        if self.config.TRAIN_DATA_PATH_PREFIX:
            return True
        else:
            return None

    def _log_predictions_during_evaluation(self, results):
        try:
            for original_name, top_predicted_words, top_scores, losses in results:
                return "{},{},{}".format(top_predicted_words[0], top_scores[0], losses) # dd for single item
        except:
            return ""

    def close_session(self):
        self.sess.close()

    def _initialize_session_variables(self):
        self.sess.run(tf.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer(),
            tf.compat.v1.tables_initializer()))
        self.log('Initalized variables')


class _TFEvaluateModelInputTensorsFormer(ModelInputTensorsFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        return (input_tensors.target_string, input_tensors.target_index,
                input_tensors.path_source_token_indices, input_tensors.path_indices,
                input_tensors.path_target_token_indices, input_tensors.context_valid_mask,
                input_tensors.path_source_token_strings, input_tensors.path_strings,
                input_tensors.path_target_token_strings)

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        return ReaderInputTensors(
            target_string=input_row[0],
            target_index=input_row[1],
            path_source_token_indices=input_row[2],
            path_indices=input_row[3],
            path_target_token_indices=input_row[4],
            context_valid_mask=input_row[5],
            path_source_token_strings=input_row[6],
            path_strings=input_row[7],
            path_target_token_strings=input_row[8]
        )
