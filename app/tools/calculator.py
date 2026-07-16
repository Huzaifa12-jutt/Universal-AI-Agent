"""Calculator tool - performs basic arithmetic safely."""

import logging
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)

_OPERATIONS = {"add", "subtract", "multiply", "divide"}


class CalculatorTool(BaseTool):
    """Performs basic arithmetic operations: add, subtract, multiply, divide."""

    name = "calculator"
    description = "Performs basic arithmetic operations (add, subtract, multiply, divide)."

    def run(self, num1: float, num2: float, operation: str) -> dict:
        logger.info("Calculator called: %s %s %s", num1, operation, num2)

        if operation not in _OPERATIONS:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of {sorted(_OPERATIONS)}."
            )

        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        else:  # divide
            if num2 == 0:
                raise ValueError("Cannot divide by zero.")
            result = num1 / num2

        return {
            "num1": num1,
            "num2": num2,
            "operation": operation,
            "result": result,
        }
