import unittest
import pandas as pd
import numpy as np

from serialize import serialize

MATCHES = 'test_data/matches.csv'
PRODUCTS = 'test_data/products.csv'
MATCHES_PRODUCTS = 'test_data/matches_products.csv'

class TestSerialize(unittest.TestCase):
    def assert_cols_length(self, sent, keep_columns):
        assert len(keep_columns) == sent.count('COL'), "Number of columns and COLs are not equal"
        assert len(keep_columns) == sent.count('VAL'), "Number of columns and VALs are not equal"

    def test_products(self):
        """Test that serialized products returns correctly"""
        products = pd.read_csv(PRODUCTS)
        serialized_matches = serialize(products=products, keep_columns=['name', 'price', 'description'])
        
        assert type(serialized_matches) == pd.DataFrame, "Wrong output type, not pd.DataFrame"
        assert np.all(np.isin(['id', 'sent'], serialized_matches.columns)), "Missing required columns: id, sent"
        assert len(serialized_matches) == len(products), "Length of input and output are not the same"
        assert np.all(np.isin(products.id.values, serialized_matches.id.values)), "Missing id values"
        self.assert_cols_length(serialized_matches.sent.iloc[0], keep_columns=['name', 'price', 'description'])
    
    def test_matches(self):
        """Test that serialized matches returns correctly"""
        matches = pd.read_csv(MATCHES)
        serialized_matches = serialize(matches=matches, keep_columns=['name', 'price'])

        assert type(serialized_matches) == pd.DataFrame, "Wrong output type, not pd.DataFrame"
        assert np.all(np.isin(['sent1', 'sent2'], serialized_matches.columns)), "Missing required columns: id, sent"
        assert len(serialized_matches) == len(matches), "Length of input and output are not the same"
        self.assert_cols_length(serialized_matches.sent1.iloc[0], keep_columns=['name', 'price'])
        self.assert_cols_length(serialized_matches.sent2.iloc[0], keep_columns=['name', 'price'])
    
    def test_product_matches(self):
        """Test that serialized matches + products returns correctly"""
        matches = pd.read_csv(MATCHES_PRODUCTS)
        products = pd.read_csv(PRODUCTS)
        serialized_matches = serialize(matches=matches, products=products, keep_columns=['name', 'price', 'description'])

        assert type(serialized_matches) == pd.DataFrame, "Wrong output type, not pd.DataFrame"
        assert np.all(np.isin(['sent1', 'sent2', 'match'], serialized_matches.columns)), "Missing required columns: id, sent"
        assert len(serialized_matches) == len(matches), "Length of input and output are not the same"
        self.assert_cols_length(serialized_matches.sent1.iloc[0], keep_columns=['name', 'price', 'description'])
        self.assert_cols_length(serialized_matches.sent2.iloc[0], keep_columns=['name', 'price', 'description'])

if __name__ == '__main__':
    unittest.main()

