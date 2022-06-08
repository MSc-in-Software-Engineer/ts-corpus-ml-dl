import os
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.word2vec import Word2Vec

from utils.base_definitions import W2V_WORKER_COUNT

MODELS_PATH = os.path.abspath('models')


class DLClassifiers:
    __w2v_model: Word2Vec
    __w2v_cbow_model: Word2Vec
    __w2v_skip_gram_model: Word2Vec

    def __init__(self, df_out: pd.DataFrame):
        self.text = df_out['text'].apply(simple_preprocess)
        self.__trained_word2vec()

    def __create_w2v_model(self, min_count: int, vector_size: int, window: int, sg: int = 0) -> Word2Vec:
        model_name: str = f'{MODELS_PATH}/w2v_cbow.model' if sg == 0 else f'{MODELS_PATH}/w2v_skip_gram.model'

        # default models find true ... load model and return
        if os.path.exists(model_name):
            model = Word2Vec.load(model_name)
            return model

        model = Word2Vec(
            min_count=min_count,
            vector_size=vector_size,
            window=window,
            sg=sg,
            workers=W2V_WORKER_COUNT
        )

        model.build_vocab(self.text, progress_per=1000)
        model.train(
            self.text,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )
        model.save(model_name)

        return model

    def __trained_word2vec(self):
        self.__w2v_cbow_model = self.__create_w2v_model(min_count=1, vector_size=100, window=5)
        self.__w2v_skip_gram_model = self.__create_w2v_model(min_count=1, vector_size=100, window=5, sg=1)

    def test_text(self, text_input: str) -> dict:
        w2v_cbow_most_similar_result = self.__w2v_cbow_model.wv.most_similar(text_input)
        w2v_skip_gram_most_similar_result = self.__w2v_skip_gram_model.wv.most_similar(text_input)

        result: dict = {
            "w2v_cbow_most_similar_result": w2v_cbow_most_similar_result,
            "w2v_skip_gram_most_similar_result": w2v_skip_gram_most_similar_result
        }

        return result
