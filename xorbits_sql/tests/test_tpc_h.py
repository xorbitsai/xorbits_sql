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

import duckdb
import pandas as pd
import pytest
from sqlglot import exp, parse_one

from .. import execute
from .helpers import FILE_DIR, TPCH_SCHEMA, load_sql

DIR = FILE_DIR + "/tpc-h/"


@pytest.fixture
def prepare_data():
    conn = duckdb.connect()

    for table, columns in TPCH_SCHEMA.items():
        conn.execute(
            f"""
                    CREATE VIEW {table} AS
                    SELECT *
                    FROM READ_CSV('{DIR}{table}.csv', delim='|', header=True, columns={columns})
                    """
        )

    sqls = [(sql, expected) for _, sql, expected in load_sql("tpc-h/tpc-h.sql")]

    try:
        yield conn, sqls
    finally:
        conn.close()


def _to_csv(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Table) and expression.name not in ("revenue"):
        return parse_one(
            f"READ_CSV('{DIR}{expression.name}.csv', 'delimiter', '|') AS {expression.alias_or_name}"
        )
    return expression


def test_execute_tpc_h(prepare_data):
    conn, sqls = prepare_data
    for sql, _ in sqls[:1]:
        expected = conn.execute(sql).fetchdf()
        result = execute(
            parse_one(sql, dialect="duckdb").transform(_to_csv).sql(pretty=True),
            TPCH_SCHEMA,
            dialect="duckdb",
        ).fetch()
        pd.testing.assert_frame_equal(result, expected)
