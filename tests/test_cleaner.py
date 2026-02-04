import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import cleaner

class TestCleaner(unittest.TestCase):
    
    def setUp(self):
        # Create dummy dataframe
        self.df = pd.DataFrame({
            'id': range(1, 6),
            'amount': [100, 200, 1000, 50, 150], # 1000 might be outlier
            'category': ['A', 'B', 'A', 'C', 'B'],
            'zipcode': ['10001', '10002', '10001', '90001', '10002'], # High cardinality potential
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-02-01', '2023-03-15', '2023-04-01']),
            'target': [0, 1, 0, 1, 0]
        })
        
        self.metadata = {
            'id': 'ID',
            'amount': 'Numerical',
            'category': 'Categorical',
            'zipcode': 'Categorical',
            'date': 'Datetime',
            'target': 'Target'
        }

    def test_datetime_extraction(self):
        options = {'drop_original_datetime': True}
        df_out = cleaner.clean_data(self.df, self.metadata, options)
        
        self.assertIn('date_month', df_out.columns)
        self.assertIn('date_day_of_week', df_out.columns)
        self.assertNotIn('date', df_out.columns)
        
        # Check specific value (2023-01-01 is Jan, month should be 1)
        self.assertEqual(df_out.iloc[0]['date_month'], 1)

    def test_scaling_standard(self):
        options = {'scaling_method': 'standard'}
        df_out = cleaner.clean_data(self.df, self.metadata, options)
        
        # Mean should be approx 0, std approx 1
        self.assertAlmostEqual(df_out['amount'].mean(), 0, places=1)
        self.assertAlmostEqual(df_out['amount'].std(), 1, places=0) # with small N, std might vary a bit if not ddof=0/1 align

    def test_encoding_onehot(self):
        # Force onehot by setting high threshold
        options = {'cardinality_threshold': 100}
        df_out = cleaner.clean_data(self.df, self.metadata, options)
        
        # Should have category_A, category_B...
        self.assertIn('category_A', df_out.columns)
        self.assertIn('category_B', df_out.columns)
        self.assertNotIn('category', df_out.columns)

    def test_encoding_frequency(self):
        # Force freq by setting low threshold
        options = {'cardinality_threshold': 2}
        df_out = cleaner.clean_data(self.df, self.metadata, options)
        
        # 'category' has 3 unique vals (A, B, C). If threshold is 2, it handles as high cardinality -> frequency
        # But 'A' appears 2/5 = 0.4. 'B' 2/5 = 0.4. 'C' 1/5 = 0.2
        self.assertIn('category', df_out.columns) # Should be kept but transformed
        self.assertTrue(np.issubdtype(df_out['category'].dtype, np.number))
        self.assertAlmostEqual(df_out.iloc[0]['category'], 0.4)

    def test_batch_id_creation(self):
        df_out = cleaner.clean_data(self.df, self.metadata, {})
        self.assertIn('Batch_ID', df_out.columns)

if __name__ == '__main__':
    unittest.main()
