import unittest
from some_utils.extract_data import date_from_filename
import datetime


class MyTestCase(unittest.TestCase):
    def test_something(self):
        path = r"G:\Mi unidad\Tesis_5\Parcela_16\16_20220215T170359_20220215T171319_T14QML"
        result = date_from_filename(path)
        print(result)
        self.assertEqual(result, datetime.date(year=2022, month=2, day=15))  # add assertion here


if __name__ == '__main__':
    unittest.main()
