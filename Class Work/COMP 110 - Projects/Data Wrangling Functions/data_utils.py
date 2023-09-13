"""Utility functions for wrangling data."""

__author__ = "730323356"


from csv import DictReader


def read_csv_rows(csv_file: str) -> list[dict[str, str]]:
    """Read a CSV file's contents into a list of rows."""
    file_handle = open(f"{csv_file}", "r", encoding="utf8")
    csv_reader = DictReader(file_handle)
    rows: list[dict[str, str]] = []
    for row in csv_reader:
        rows.append(row)
    file_handle.close()
    return rows


def column_values(a: list[dict[str, str]], b: str) -> list[str]:
    """Returns a list of values in a single column."""
    column_values: list[str] = []
    for row in a:
        column_values.append(row[b])
    return column_values


def columnar(a: list[dict[str, str]]) -> dict[str, list[str]]:
    """Transforms a table as a list of rows into a table as a list of columns."""
    table: dict[str, list[str]] = {}
    keys: list[str] = []
    temp: dict[str, str] = {}
    for row in a:
        temp = row
        keys = list(temp.keys())
    counter: int = 0
    while counter < len(keys):
        table[keys[counter]] = column_values(a, keys[counter])
        counter += 1
    return table


def head(a: dict[str, list[str]], b: int) -> dict[str, list[str]]:
    """Produces a column-based table with first N rows of data."""
    temp: dict[str, list[str]] = {}
    cols: list[str] = list(a.keys())
    for column in cols:
        tempa: list[str] = []
        row_vals: list[str] = []
        counter: int = 0
        test: int = len(list(a[column]))
        while counter < b and counter < test:
            row_vals = list(a[column])
            tempa.append(row_vals[counter])
            counter += 1
        temp[column] = tempa
    return temp


def select(a: dict[str, list[str]], b: list[str]) -> dict[str, list[str]]:
    """Produces a column-based table with specific columns from the original."""
    temp: dict[str, list[str]] = {}
    for column in b:
        temp[column] = a[column]
    return temp


def count(a: list[str]) -> dict[str, int]:
    """Returns dictionary where each key has an associated value that indicates its frequency."""
    temp: dict[str, int] = {}
    for val in a:
        if val in temp:
            temp[val] += 1
        else:
            temp[val] = 1
    return temp