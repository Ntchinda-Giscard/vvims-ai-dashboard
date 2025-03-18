import unittest


class Calculator:
    def add(self, a, b):
        return a + b

class TestCalculator(unittest.TestCase):

    def test_add(self):
        calc = Calculator()
        self.assertEqual(calc.add(2, 3), 5)
        self.assertEqual(calc.add(-1, 1), 0)
        self.assertEqual(calc.add(0, 0), 0)

if __name__ == "__main__":
    unittest.main()