from typing import Type
from typing import TypeVar

class ArgsValidator:

    T = TypeVar("T")

    def require_type_not_none(value: object, expected_type: Type[T], description: str) -> T:
        if value is None:
            raise ValueError(f"{description} cannot be None")

        if not isinstance(value, expected_type):
            raise TypeError(
                f"{description} must be of type {expected_type.__name__}, "
                f"but got {type(value).__name__}"
            )
        return value
