import unittest

from src.common.validator.args_validator import ArgsValidator


class TestArgsValidator(unittest.TestCase):

    def test_require_type_not_none_returns_value_when_str_type_matches(self):
        value = "hello"

        result = ArgsValidator.require_type_not_none(value, str, "name")

        self.assertEqual("hello", result)

    def test_require_type_not_none_returns_value_when_list_type_matches(self):
        value = [1, 2, 3]

        result = ArgsValidator.require_type_not_none(value, list, "items")

        self.assertEqual(value, result)

    def test_require_type_not_none_raises_value_error_when_value_is_none(self):
        with self.assertRaises(ValueError) as context:
            ArgsValidator.require_type_not_none(None, str, "username")

        self.assertEqual("username cannot be None", str(context.exception))

    def test_require_type_not_none_raises_type_error_when_type_is_wrong(self):
        with self.assertRaises(TypeError) as context:
            ArgsValidator.require_type_not_none("123", int, "age")

        self.assertEqual("age must be of type int, but got str", str(context.exception))

    def test_require_type_not_none_with_int_rejects_float(self):
        with self.assertRaises(TypeError) as context:
            ArgsValidator.require_type_not_none(1.5, int, "count")

        self.assertEqual("count must be of type int, but got float",  str(context.exception))

    def test_require_type_not_none_accepts_subclass_instance_but_not_parent(self):
        class Animal:
            pass

        class Dog(Animal):
            pass

        value = Dog()

        result = ArgsValidator.require_type_not_none(value, Animal, "sub type")

        self.assertIs(value, result)

        with self.assertRaises(TypeError) as context:
            ArgsValidator.require_type_not_none(Animal, Dog, "super type")

        self.assertEqual('super type must be of type Dog, but got type', str(context.exception))
