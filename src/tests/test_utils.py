from unittest import TestCase
from src.utils import *


class TestUtils(TestCase):

    def test_load_input(self):
        files, train, predict = load_input('input_test.csv')
        self.assertEqual(type(files), dict)
        self.assertEqual(train.shape, (455401, 24))
        self.assertEqual(predict.shape, (151655, 23))

    def test_perform_undersampling(self):
        X_train, X_test, y_train, y_test = perform_undersampling(pd.read_csv('../../data/training_sample.csv', index_col=[0]))
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])

    def test_train_models(self):
        X_train, X_test, y_train, y_test = perform_undersampling(pd.read_csv('../../data/training_sample.csv', index_col=[0]))
        trained_models, voting_model = train_models(X_train, X_test, y_train, y_test)

    def test_make_prediction(self):
        X_train, X_test, y_train, y_test = perform_undersampling(
            pd.read_csv('../../data/training_sample.csv', index_col=[0]))
        trained_models, voting_model = train_models(X_train, X_test, y_train, y_test)
        make_prediction(pd.read_csv('../../data/prediction_input.csv', index_col=[0]), 'results.csv', trained_models, voting_model)
