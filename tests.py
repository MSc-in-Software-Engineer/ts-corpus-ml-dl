import time

from ai_concepts.deep_learning.dl_classifiers import DLClassifiers
from ai_concepts.machine_learning.ml_classifiers import MLClassifiers
from src.text_processing_wizard import TextProcessingWizard

if __name__ == '__main__':
    text_processing_wizard = TextProcessingWizard()

    start_time = time.perf_counter()
    ml_classifiers = MLClassifiers(df_out=text_processing_wizard.get_dataframe())
    print(ml_classifiers.call_ml_classifiers())

    dl_classifiers = DLClassifiers(df_out=text_processing_wizard.get_dataframe())
    print(dl_classifiers.test_text('bilim'))
    finish_time = time.perf_counter()

    print(f"Finished in {round(finish_time - start_time, 3)} seconds")
