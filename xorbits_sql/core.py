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

import logging
import time
from typing import TYPE_CHECKING

import xorbits
import xorbits.pandas as pd
from sqlglot.optimizer import optimize
from sqlglot.planner import Plan
from sqlglot.schema import (
    dict_depth,
    ensure_schema,
    flatten_schema,
    nested_get,
    nested_set,
)

from .errors import ExecuteError
from .executor import XorbitsExecutor
from .table import Table, ensure_tables

if TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType
    from sqlglot.expressions import Expression
    from sqlglot.schema import Schema

logger = logging.getLogger(__name__)


PYTHON_TYPE_TO_SQLGLOT = {
    "dict": "MAP",
}


def execute(
    sql: str | Expression,
    schema: dict | Schema | None = None,
    dialect: DialectType = None,
    tables: dict[str, pd.DataFrame | Table] | None = None,
) -> pd.DataFrame:
    """
    Run a sql query against data.

    Args:
        sql: a sql statement.
        schema: database schema.
            This can either be an instance of `Schema` or a mapping in one of the following forms:
            1. {table: {col: type}}
            2. {db: {table: {col: type}}}
            3. {catalog: {db: {table: {col: type}}}}
        dialect: the SQL dialect to apply during parsing (eg. "spark", "hive", "presto", "mysql").
        tables: additional tables to register.

    Returns:
        Table that represents data.
    """
    tables_ = ensure_tables(tables, dialect=dialect)
    xorbits.run([t.df for t in tables_.mapping.values()])

    if not schema:
        schema = {}
        flattened_tables = flatten_schema(
            tables_.mapping, depth=dict_depth(tables_.mapping)
        )

        for keys in flattened_tables:
            table = nested_get(tables_.mapping, *zip(keys, keys))
            assert table is not None

            for column in table.columns:
                py_type = type(table[0][column]).__name__
                nested_set(
                    schema,
                    [*keys, column],
                    PYTHON_TYPE_TO_SQLGLOT.get(py_type) or py_type,
                )

    schema = ensure_schema(schema, dialect=dialect)

    if (
        tables_.supported_table_args
        and tables_.supported_table_args != schema.supported_table_args
    ):
        raise ExecuteError("Tables must support the same table args as schema")

    now = time.time()
    expression = optimize(sql, schema, leave_tables_isolated=True, dialect=dialect)

    logger.debug("Optimization finished: %f", time.time() - now)
    logger.debug("Optimized SQL: %s", expression.sql(pretty=True))

    plan = Plan(expression)

    logger.debug("Logical Plan: %s", plan)

    now = time.time()
    result = XorbitsExecutor(tables=tables_).execute(plan)

    logger.debug("Query finished: %f", time.time() - now)

    return result
