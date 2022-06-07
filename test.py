import time

from ai_concepts.machine_learning.ml_classifiers import MLClassifiers
from src.text_processing_wizard import TextProcessingWizard

if __name__ == '__main__':
    text_processing_wizard = TextProcessingWizard()

    start_time = time.perf_counter()
    ml_classifiers = MLClassifiers(df_out=text_processing_wizard.get_dataframe())
    print(ml_classifiers.call_ml_classifiers())
    finish_time = time.perf_counter()

    print(f"Finished in {round(finish_time - start_time, 3)} seconds")
