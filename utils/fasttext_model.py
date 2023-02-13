import os
import pickle
import logging
import numpy as np

from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec


class ModelManager:
#     models_folder = os.path.join(parent_folder, "models")

    def __init__(self, config):
        self.config = config

    class Callback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0
            self.loss_previous_step = 0

        def on_epoch_end(self, model):
            try:
                loss = model.get_latest_training_loss()
            except Exception as e:
                loss = model.running_training_loss
            if self.epoch == 0:
                logging.info('Loss após a época {}: {}'.format(self.epoch, loss))
            else:
                logging.info('Loss após a época {}: {}'.format(self.epoch, loss - self.loss_previous_step))
            self.epoch += 1
            self.loss_previous_step = loss

    def train_model(self, list_tokens):
        logging.info(f"Training with FastText algorithm")
        return self._train_model_fast_text(list_tokens)

    def _train_model_fast_text(self, list_tokens):
        fsttxt_model = FastText(sg=self.config.get('sg', 0),
                                window=self.config.get('window', 2),
                                size=self.config.get('size', 200),
                                alpha=self.config.get('alpha', 0.03),
                                word_ngrams=self.config.get('word_ngrams', 2),
                                min_count=self.config.get('min_count', 2),
                                min_alpha=self.config.get('min_alpha', 0.007),
                                hs=self.config.get('hs', 1))
        fsttxt_model.build_vocab(list_tokens, progress_per=100)
        logging.info("Built the model's vocabulary")
        fsttxt_model.train(list_tokens,
                           total_examples=fsttxt_model.corpus_count,
                           total_words=fsttxt_model.corpus_total_words,
                           epochs=self.config.get('epochs', 100),
                           loss='softmax',
                           compute_loss=self.config.get('compute_loss', True),
                           callbacks=[self.Callback()])
        logging.info("Trained the model completely")
        
        with open(f"model/{self.config.get('model_name')}.pkl", "wb") as f:
            pickle.dump(fsttxt_model, f)
        logging.info(f"Saved the model.")

    def load_model(self):
        try:
            logging.info(f"Looking for model {self.config.get('model_name')} in model folder...")
            model = FastText.load(f"model/{self.config.get('model_name')}.pkl")
            logging.info(f"Loaded as FastText model!")

            return model
        except KeyError:
            pass

    def vector_combination_sum(self, words):
        result = np.zeros(self.config.get('fast_text').get('size'))
        model = self.load_model()
        for wrd in words:
            try:
                result += model.wv.get_vector(wrd)
            except KeyError:
                pass

        return result

    def vector_matrix_embedding(self, corpus):
        x = len(corpus)
        y = self.config.get('fast_text').get('size')  # colunas/dimensões
        matrix = np.zeros((x, y))
        logging.info(f"Creating vector matrix {x} x {y}...")
        for i in range(x):
            matrix[i] = self.vector_combination_sum(corpus[i])

        return matrix


