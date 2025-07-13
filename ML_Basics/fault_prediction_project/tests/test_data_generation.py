# tests/test_data_generation.py
import unittest
from src.data_generation import generate_system_metrics

class TestDataGeneration(unittest.TestCase):
    def test_data_shape(self):
        data = generate_system_metrics(n_samples=100)
        self.assertEqual(len(data), 100)
        self.assertEqual(set(data.columns), {'timestamp', 'device_id', 'cpu_usage', 'ram_usage', 'disk_io', 'temperature', 'error_count', 'label'})

if __name__ == '__main__':
    unittest.main()
