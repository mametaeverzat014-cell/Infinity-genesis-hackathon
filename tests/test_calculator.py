import unittest

from auditgym.calculator import CalcError, evaluate


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(evaluate("1+2*3"), "7")
        self.assertEqual(evaluate("(1250 * 1.0842)"), "1355.25")
        self.assertEqual(evaluate("10/4"), "2.5")
        self.assertEqual(evaluate("-3 + +2"), "-1")
        self.assertEqual(evaluate(" 0.1 + 0.2 "), "0.3")
        self.assertEqual(evaluate("123456789 * 0.00210437"), "259798.76306793")

    def test_rejects_code_and_names(self):
        for bad in ["__import__('os')", "a+1", "open('x')", "2**10", "1e5", "1;2", "[1]", "1 if 1 else 2",
                    "lambda: 1", "'1'+'2'", "1,2", "abs(-1)", "x.y", ""]:
            with self.assertRaises(CalcError, msg=bad):
                evaluate(bad)

    def test_limits(self):
        with self.assertRaises(CalcError):
            evaluate("1/0")
        with self.assertRaises(CalcError):
            evaluate("1+" * 200 + "1")
        with self.assertRaises(CalcError):
            evaluate("(" * 60 + "1" + ")" * 60)
        with self.assertRaises(CalcError):
            evaluate(None)
        with self.assertRaises(CalcError):
            evaluate("(1+2")


if __name__ == "__main__":
    unittest.main()
