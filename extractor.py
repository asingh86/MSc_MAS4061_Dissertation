import common
import os
import pandas as pd
import requests
import gzip
import random
from typing import BinaryIO


class DataExtractor:

    def __init__(self):
        self._config = common.read_configs()

    def get_filepath(self) -> str:
        file_name = self._config['data_url'].split("/")[-1].replace('.', "_") + '.gz'
        file_path = os.path.join(os.getcwd(), 'data/', file_name)
        return file_path

    def download_data(self) -> BinaryIO:
        """
        Download data from the url
        :return:
        """
        with open(self.get_filepath(), "wb") as f:
            r = requests.get(self._config['data_url'])
            f.write(r.content)
            f.close()

    def get_samples(self, sample_size: float, seed: int) -> pd.DataFrame:
        """
        creates a dataframe of random sample from the file
        :param sample_size: probability of selecting a sample and should be between 0 and 100
        :param seed: This is to assign random seed for reproducability
        :return: returns a pandas dataframe
        """
        if 0 < sample_size > 100:
            raise Exception("Sample size need to be between 0 and 100")

        with gzip.open(self.get_filepath(), 'rt', encoding="utf8") as f:
            header = f.readline()
            header = header.strip().split('\t')
            sample_df = pd.DataFrame(columns=header)
            random.seed(seed)
            for line in f:
                sample = random.choices([0, 1], weights=[100 - sample_size, sample_size], k=1)
                if sample[0] == 1:

                    fields = line.split('\t')
                    fields = pd.Series(fields, index=header)
                    sample_df = sample_df.append(fields, ignore_index=True)
        return sample_df


    def get_data(self, sample_size: float = 0.01, seed: int = 123) -> pd.DataFrame():
        """
        read or download the data file
        :param sample_size: probability of selecting a sample and should be between 0 and 100
        :param seed: This is to assign random seed for reproducability
        :return: returns a pandas dataframe
        """

        if os.path.exists(self.get_filepath()):
            df = self.get_samples(sample_size, seed)
        else:
            self.download_data()
            df = self.get_samples(sample_size, seed)
        return df

    def split_train_test_samples(sample_size: float, seed: int) -> pd.DataFrame:
        """
        creates a dataframe of random sample from the file
        :param sample_size: probability of selecting a sample and should be between 0 and 100
        :param seed: This is to assign random seed for reproducability
        :return: returns a pandas dataframe
        """
        # if sample_size<0 | sample_size> 100:
        #   raise Exception("Sample size need to be between 0 and 100")

        header_filepath = os.path.join(os.getcwd(), 'data/header.csv')
        train_pos_filepath = os.path.join(os.getcwd(), 'data/train_pos.csv')
        train_neg_filepath = os.path.join(os.getcwd(), 'data/train_neg.csv')
        test_pos_filepath = os.path.join(os.getcwd(), 'data/test_pos.csv')
        test_neg_filepath = os.path.join(os.getcwd(), 'data/test_neg.csv')
        combined_filepath = [header_filepath, train_pos_filepath, train_neg_filepath, test_pos_filepath,
                             test_neg_filepath]

        for path in combined_filepath:
            if os.path.exists(path):
                os.remove(path)

        with gzip.open(filepath, 'rt', encoding="utf8") as f, \
                open(header_filepath, 'w') as header, \
                open(train_pos_filepath, 'w') as train_pos, \
                open(train_neg_filepath, 'w') as train_neg, \
                open(test_pos_filepath, 'w') as test_pos, \
                open(test_neg_filepath, 'w') as test_neg:

            header.write(f.readline())
            random.seed(seed)
            for line in f:
                sample = random.choices([0, 1], weights=[100 - sample_size, sample_size], k=1)
                fields = line.split('\t')
                polarity = -1 if int(fields[7]) < 3 else 1 if int(fields[7]) > 3 else 0
                if sample[0] == 1:
                    if polarity == 1:
                        train_pos.write(line)
                    elif polarity == -1:
                        train_neg.write(line)
                else:
                    if polarity == 1:
                        test_pos.write(line)
                    elif polarity == -1:
                        test_neg.write(line)


