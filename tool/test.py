"""Test Module."""
from tool import *
import pandas as pd
import pytest


class TestStandardization(object):
    """
    Class for testing the time standardisation and its functions.
    """

    @pytest.mark.parametrize('dataset1, new_dataset1, dataset2,\
                             new_dataset2', [
        (((1, 2), ('p', 'pppp'), ('qiiq', 'qqq')),
         (('oo', '99'), (1, 2), ('p', 'pppp'), ('qiiq', 'qqq')),
         ((1, 2), (23, 0), ('wp', 'qq')),
         ((1, 2), (23, 0), ('oo', '99'), ('wp', 'qq'))
         )
    ])
    def test_insert_point(self, dataset1, new_dataset1,
                          dataset2, new_dataset2):
        """ Test the insert_point function """

        test_set1 = insert_point(0, dataset1, ('oo', '99'))
        test_set2 = insert_point(2, dataset2, ('oo', '99'))

        for i, j in zip(test_set1, new_dataset1):
            assert i == j

        for i, j in zip(test_set2, new_dataset2):
            assert i == j

    @pytest.mark.parametrize('df1, df2, new_df1, new_df2', [
        (pd.DataFrame([['tom', 10], ['nick', 15], ['juli', 14]],
                      columns=['Name', 'Age']),
         pd.DataFrame([['tom', 10], ['nick', 15], ['juli', 14]],
                      columns=['Name', 'Age']),
         pd.DataFrame([['bob', 25], ['tom', 10], ['nick', 15],
                      ['juli', 14]], columns=['Name', 'Age']),
         pd.DataFrame([['tom', 10], ['bob', 25], ['nick', 15],
                      ['juli', 14]], columns=['Name', 'Age'])
         )
    ])
    def test_Insert_row(self, df1, df2, new_df1, new_df2):
        """ Test the Insert_row function """

        test_df1 = Insert_row(0, df1, ['bob', 25])
        test_df2 = Insert_row(1, df2, ['bob', 25])

        assert test_df1.equals(new_df1)
        assert test_df2.equals(new_df2)

    @pytest.mark.parametrize('df1, new_df1, dataset, new_dataset', [
        (pd.DataFrame([['fcr_000', 'fcr', '0', '1', '25'],
                      ['fcr_001', 'fcr', '1800', '1', '25'],
                      ['fcr_002', 'fcr', '5400', '1', '25']],
                      columns=['Image ID', 'Storm ID', 'Relative Time',
                      'Ocean', 'Wind Speed']),
         pd.DataFrame([['fcr_000', 'fcr', '0', '1', '25'],
                      ['fcr_001', 'fcr', '1800', '1', '25'],
                      ['fcr_fake_1point5', 'fcr', '3600', '1', '25'],
                      ['fcr_002', 'fcr', '5400', '1', '25']],
                      columns=['Image ID', 'Storm ID', 'Relative Time',
                      'Ocean', 'Wind Speed']),
         ((1, 1), (2, 2), (3, 3)), ((1, 1), (2, 2), (2.5, 2.5), (3, 3))
         )
    ])
    def test_standardization(self, df1, new_df1, dataset, new_dataset):
        test_df1, test_dataset = standardise(df1, dataset)

        assert test_df1.equals(new_df1)
        for i, j in zip(test_dataset, new_dataset):
            assert i == j

    str1 = 'data/nasa_tropical_storm_competition_train_source/'
    str2 = 'nasa_tropical_storm_competition_train_source_fcr_000/image.jpg'
    str3 = 'data/nasa_tropical_storm_competition_train_source/nasa_tropical'
    str4 = '_storm_competition_train_source_fcr_001/image.jpg'

    @pytest.mark.parametrize('df1, new_df1, dataset', [
        (pd.DataFrame([['fcr_000', 'fcr', '0', '1', '25'], ['fcr_001',
                       'fcr', '1800', '1', '25']],
                      columns=['Image ID', 'Storm ID', 'Relative Time',
                      'Ocean', 'Wind Speed']),
         pd.DataFrame([['fcr_000', 'fcr', '0', '1', '25', str1 + str2],
                       ['fcr_001', 'fcr', '1800', '1', '25', str3 + str4]],
                      columns=['Image ID', 'Storm ID', 'Relative Time',
                               'Ocean', 'Wind Speed', 'Links']),
         ((1, 2), (1, 2))
         )
    ])
    def test_links(self, df1, new_df1, dataset):
        """ Test the link """

        test_df1 = add_links(df1, dataset)

        assert test_df1.equals(new_df1)
