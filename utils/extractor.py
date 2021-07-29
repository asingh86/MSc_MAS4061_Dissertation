from utils import common
import os
import pandas as pd
import requests
import gzip
import random
from typing import BinaryIO
import csv
import numpy as np


class DataExtractor:

    def __init__(self):
        self._config = common.read_configs()

    def get_filepath(self) -> list:
        """
        :return: list of file paths
        """
        file_name = self._config['data_url'].split("/")[-1].replace('.', "_") + '.gz'
        file_path = os.path.join(os.getcwd(), 'data/', file_name)
        header_filepath = os.path.join(os.getcwd(), self._config['files']['header'])
        train_pos_filepath = os.path.join(os.getcwd(), self._config['files']['train_pos'])
        train_neg_filepath = os.path.join(os.getcwd(), self._config['files']['train_neg'])
        test_pos_filepath = os.path.join(os.getcwd(), self._config['files']['test_pos'])
        test_neg_filepath = os.path.join(os.getcwd(), self._config['files']['test_neg'])
        return [file_path, header_filepath, train_pos_filepath, train_neg_filepath,
                test_pos_filepath, test_neg_filepath]

    def download_data(self) -> BinaryIO:
        """
        Download data from the url and save it in a file
        """
        with open(self.get_filepath()[0], "wb") as f:
            r = requests.get(self._config['data_url'])
            f.write(r.content)
            f.close()

    def count_rows(self) -> int:
        """
        Counts the number of records in a file excluding header
        """

        combined_filepath = self.get_filepath()
        records = -1  # this is to ensure we skip header
        with gzip.open(combined_filepath[0], 'rt', encoding="utf8") as f:
            for line in f:
                records += 1
        return records

    def split_train_test_samples(self, sample_size: float, seed: int):
        """
        creates a csv file and stores in the location specified
        :param sample_size: probability of selecting a sample and should be between 0 and 100
        :param seed: This is to assign random seed for reproducability
        :return: returns a pandas dataframe
        """
        # if sample_size<0 | sample_size> 100:
        #   raise Exception("Sample size need to be between 0 and 100")

        combined_filepath = self.get_filepath()

        with gzip.open(combined_filepath[0], 'rt', encoding="utf8") as f, \
                open(combined_filepath[1], 'w') as header, \
                open(combined_filepath[2], 'w') as train_pos, \
                open(combined_filepath[3], 'w') as train_neg, \
                open(combined_filepath[4], 'w') as test_pos, \
                open(combined_filepath[5], 'w') as test_neg:

            header_writer = csv.writer(header, delimiter='\t', lineterminator='\n')
            train_pos_writer = csv.writer(train_pos, delimiter='\t', lineterminator='\n')
            train_neg_writer = csv.writer(train_neg, delimiter='\t', lineterminator='\n')
            test_pos_writer = csv.writer(test_pos, delimiter='\t', lineterminator='\n')
            test_neg_writer = csv.writer(test_neg, delimiter='\t', lineterminator='\n')

            header_writer.writerow(f.readline().split('\t'))
            random.seed(seed)

            for line in f:
                fields = line.split('\t')
                polarity = -1 if int(fields[7]) < 3 else 1 if int(fields[7]) > 3 else 0
                sample = random.choices([0, 1], weights=[100 - sample_size, sample_size], k=1)
                if sample[0] == 1:
                    if polarity == 1:
                        train_pos_writer.writerow(fields)
                    elif polarity == -1:
                        train_neg_writer.writerow(fields)
                else:
                    if polarity == 1:
                        test_pos_writer.writerow(fields)
                    elif polarity == -1:
                        test_neg_writer.writerow(fields)

    # ToDo: split this function so it can read and output individual file
    def get_data(self, sample_size: float = 70, seed: int = 123) -> {str: pd.DataFrame}:
        """
        read or download the data file
        :param sample_size: probability of selecting a sample and should be between 0 and 100
        :param seed: This is to assign random seed for reproducability
        :return: returns a dictionary of sampled dataframes
        """

        combined_filepath = self.get_filepath()

        if not os.path.exists(combined_filepath[0]):
            self.download_data()
            self.split_train_test_samples(sample_size, seed)
        else:
            for path in combined_filepath[1:]:
                if not os.path.exists(path):
                    self.split_train_test_samples(sample_size, seed)

        files = {}
        for path in combined_filepath[1:]:
            filename = os.path.basename(path).split('.')[0]
            files[filename] = pd.read_csv(path, header=None, sep="\t", error_bad_lines=False)

        return files

    def delete_data(self):
        """
        This method deletes any existing files in the filepath
        """

        combined_filepath = self.get_filepath()
        for file_path in combined_filepath:
            if os.path.exists(file_path):
                os.remove(file_path)

    def process_freq_text(self):
        files = self.get_data()

        train_reviews = files['train_neg'][13].to_list() + files['train_pos'][13].to_list()
        train_labels = np.append(np.ones((len(files['train_neg']))), np.zeros((len(files['train_pos']))))
        train_labels = train_labels.tolist()

        test_reviews = files['test_neg'][13].to_list() + files['test_pos'][13].to_list()
        test_labels = np.append(np.ones((len(files['test_neg']))), np.zeros((len(files['test_pos']))))
        test_labels = test_labels.tolist()

        return train_reviews, train_labels, test_reviews, test_labels
