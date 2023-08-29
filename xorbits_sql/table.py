# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import xorbits.pandas as pd
from sqlglot.dialects.dialect import DialectType
from sqlglot.schema import AbstractMappingSchema, normalize_name


class Table:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = tuple(df.dtypes.index)

    def __getitem__(self, index: int):
        return self.df.iloc[index].fetch().to_dict()


class Tables(AbstractMappingSchema[Table]):
    pass


def ensure_tables(d: dict | None, dialect: DialectType = None) -> Tables:
    return Tables(_ensure_tables(d, dialect=dialect))


def dict_depth(d: dict) -> int:
    """
    Get the nesting depth of a dictionary.

    Example:
        >>> dict_depth(None)
        0
        >>> dict_depth({})
        1
        >>> dict_depth({"a": "b"})
        1
        >>> dict_depth({"a": {}})
        2
        >>> dict_depth({"a": {"b": {}}})
        3
    """
    try:
        if isinstance(d, (pd.DataFrame, pd.Series)):
            raise AttributeError
        return 1 + dict_depth(next(iter(d.values())))
    except AttributeError:
        # d doesn't have attribute "values"
        return 0
    except StopIteration:
        # d.values() returns an empty sequence
        return 1


def _ensure_tables(
    d: dict[str, pd.DataFrame | Table | dict] | Tables | None,
    dialect: DialectType = None,
) -> dict[str, Table]:
    if not d:
        return {}

    depth = dict_depth(d)
    if depth > 1:
        return {
            normalize_name(k, dialect=dialect, is_table=True): _ensure_tables(  # type: ignore
                v, dialect=dialect  # type: ignore
            )
            for k, v in d.items()
        }

    result = {}
    for table_name, table in d.items():
        table_name = normalize_name(table_name, dialect=dialect)

        if isinstance(table, Table):
            result[table_name] = table
        elif isinstance(table, pd.DataFrame):
            result[table_name] = Table(table)
        else:
            table = [
                {
                    normalize_name(column_name, dialect=dialect): value
                    for column_name, value in row.items()
                }
                for row in table
            ]
            column_names = (
                tuple(column_name for column_name in table[0]) if table else ()
            )
            rows = [tuple(row[name] for name in column_names) for row in table]
            result[table_name] = Table(pd.DataFrame(rows, columns=column_names))

    return result
