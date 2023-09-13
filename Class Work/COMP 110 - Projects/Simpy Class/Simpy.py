"""Utility class for numerical operations."""

from __future__ import annotations

from typing import Union

__author__ = "730323356" 


class Simpy:
    """This is the Simpy class."""
    values: list[float]

    def __init__(self, values: list[float]):
        """Constructor."""
        self.values = values

    def __repr__(self) -> str:
        """Print string representation."""
        return f"Simpy({self.values})"

    def fill(self, new: float, length: int) -> None:
        """Creates list of the same value new, length times."""
        i: int = 0       
        if length > len(self.values):
            while i < length:
                if i < len(self.values):
                    self.values[i] = new
                else:
                    self.values.append(new)
                i += 1
        else:
            while i < len(self.values):
                if i < length:
                    self.values[i] = new
                else: 
                    self.values.remove(self.values[i])
                i += 1

    def arange(self, start: float, stop: float, step: float = 1.0) -> None:
        """Creates list ranging from start, to on eless than stop, with step size step."""
        counter: float = start
        length: float = stop - start
        num_steps: int = length / step
        i: int = 0
        while i < num_steps:
            if i < len(self.values):
                self.values[i] = counter
            else:
                self.values.append(counter)
            counter += step
            i += 1

    def sum(self) -> float:
        """Sums values in list and returns them."""
        length: int = len(self.values)
        i: int = 0
        temp: float = 0.0
        while i < length:
            temp += self.values[i]
            i += 1
        return temp

    def __add__(self, rhs: Union[float, Simpy]) -> Simpy:
        """Addition overload."""
        result: list[float] = []
        if isinstance(rhs, float):
            for item in self.values:
                result.append(item + rhs)
        else: 
            assert len(self.values) == len(rhs.values)
            for i in range(len(self.values)):
                result.append(self.values[i] + rhs.values[i])
        return Simpy(result)

    def __pow__(self, rhs: Union[float, Simpy]) -> Simpy:
        """Power overload."""
        result: list[float] = []
        if isinstance(rhs, float):
            for item in self.values:
                result.append(item ** rhs)
        else: 
            assert len(self.values) == len(rhs.values)
            for i in range(len(self.values)):
                result.append(self.values[i] ** rhs.values[i])
        return Simpy(result)

    def __mod__(self, rhs: Union[float, Simpy]) -> Simpy:
        """Modulus overload."""
        result: list[float] = []
        if isinstance(rhs, float):
            for item in self.values:
                result.append(item % rhs)
        else: 
            assert len(self.values) == len(rhs.values)
            for i in range(len(self.values)):
                result.append(self.values[i] % rhs.values[i])
        return Simpy(result)

    def __eq__(self, rhs: Union[float, Simpy]) -> list[bool]:
        """Equality overload."""
        result: list[bool] = []
        if isinstance(rhs, float):
            for item in self.values:
                result.append(item == rhs)
        else: 
            assert len(self.values) == len(rhs.values)
            for i in range(len(self.values)):
                result.append(self.values[i] == rhs.values[i])
        return result

    def __gt__(self, rhs: Union[float, Simpy]) -> list[bool]:
        """Greater than overload."""
        result: list[bool] = []
        if isinstance(rhs, float):
            for item in self.values:
                result.append(item > rhs)
        else: 
            assert len(self.values) == len(rhs.values)
            for i in range(len(self.values)):
                result.append(self.values[i] > rhs.values[i])
        return result

    def __getitem__(self, rhs: Union[int, list[bool]]) -> Union[float, Simpy]:
        """Getitem overload."""
        if isinstance(rhs, int):
            result: float
            result = self.values[rhs]
            return result
        else:
            result: list[bool] = []
            length: int = len(self.values)
            i: int = 0
            while i < length:
                if rhs[i]:
                    result.append(self.values[i])
                i += 1
            return Simpy(result)