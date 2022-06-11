import os
import glob
import codecs
import pandas as pd

from utils.base_definitions import (
    DATASETS_PATH,
    TTC_3600_DIR,
    ALPHA_STRING_VALUES,
    UNWANTED_STRING_VALUES
)

DEFAULT_DATASETS_PATH: str = f"{DATASETS_PATH}/{TTC_3600_DIR}"


class TextProcessingWizard:
    __datasets_path: str = ''
    __data_list: list = []
    __label_list: list = []
    __df_out: pd.DataFrame = pd.DataFrame()

    def __init__(self, datasets_path: str = DEFAULT_DATASETS_PATH):
        self.__datasets_path = os.path.abspath(datasets_path)
        self.dirs = os.listdir(self.__datasets_path)

        self.__txt_collection_merge()
        self.__df_out['label'] = self.__label_list
        self.__df_out['text'] = self.__data_list

    @staticmethod
    def __read_text(file_name: str) -> list:
        fp = codecs.open(filename=file_name, encoding='utf-8')
        lines = fp.readlines()
        return lines

    @staticmethod
    def __clean_text(lines: list) -> str:
        lines = [x.lower().strip() for x in lines]
        txt = ' '.join(lines)

        for uw in UNWANTED_STRING_VALUES:
            txt = txt.replace(uw, ' ')
        txt = txt.replace(u'â', 'a')
        while '  ' in txt:
            txt = txt.replace('  ', ' ')

        chars = list(set(txt))
        for ch in chars:
            if not ch in ALPHA_STRING_VALUES:
                txt = txt.replace(ch, '')

        return txt.strip()

    def __txt_collection_merge(self):
        for data_name in self.dirs:
            file_names = glob.glob('{path}/{data_name}/*.txt'.format(
                path=self.__datasets_path,
                data_name=data_name)
            )
            file_names = [f.replace('\\', '/') for f in file_names]

            for file_name in file_names:
                lines = self.__read_text(file_name=file_name)
                txt = self.__clean_text(lines=lines)

                self.__label_list.append(data_name)
                self.__data_list.append(txt)

    def get_dataframe(self):
        return self.__df_out
