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

import os

FILE_DIR = os.path.dirname(__file__)


def _filter_comments(s: str) -> str:
    return "\n".join(
        [line for line in s.splitlines() if line and not line.startswith("--")]
    )


def _extract_meta(sql: str) -> tuple[str, dict[str, str]]:
    meta = {}
    sql_lines = sql.split("\n")
    i = 0
    while sql_lines[i].startswith("#"):
        key, val = sql_lines[i].split(":", maxsplit=1)
        meta[key.lstrip("#").strip()] = val.strip()
        i += 1
    sql = "\n".join(sql_lines[i:])
    return sql, meta


def load_sql(filename: str):
    with open(os.path.join(FILE_DIR, filename), encoding="utf-8") as f:
        statements = _filter_comments(f.read()).split(";")

        size = len(statements)

        for i in range(0, size, 2):
            if i + 1 < size:
                sql = statements[i].strip()
                sql, meta = _extract_meta(sql)
                expected = statements[i + 1].strip()
                yield meta, sql, expected


TPCH_SCHEMA = {
    "lineitem": {
        "l_orderkey": "bigint",
        "l_partkey": "bigint",
        "l_suppkey": "bigint",
        "l_linenumber": "bigint",
        "l_quantity": "double",
        "l_extendedprice": "double",
        "l_discount": "double",
        "l_tax": "double",
        "l_returnflag": "string",
        "l_linestatus": "string",
        "l_shipdate": "string",
        "l_commitdate": "string",
        "l_receiptdate": "string",
        "l_shipinstruct": "string",
        "l_shipmode": "string",
        "l_comment": "string",
    },
    "orders": {
        "o_orderkey": "bigint",
        "o_custkey": "bigint",
        "o_orderstatus": "string",
        "o_totalprice": "double",
        "o_orderdate": "string",
        "o_orderpriority": "string",
        "o_clerk": "string",
        "o_shippriority": "int",
        "o_comment": "string",
    },
    "customer": {
        "c_custkey": "bigint",
        "c_name": "string",
        "c_address": "string",
        "c_nationkey": "bigint",
        "c_phone": "string",
        "c_acctbal": "double",
        "c_mktsegment": "string",
        "c_comment": "string",
    },
    "part": {
        "p_partkey": "bigint",
        "p_name": "string",
        "p_mfgr": "string",
        "p_brand": "string",
        "p_type": "string",
        "p_size": "int",
        "p_container": "string",
        "p_retailprice": "double",
        "p_comment": "string",
    },
    "supplier": {
        "s_suppkey": "bigint",
        "s_name": "string",
        "s_address": "string",
        "s_nationkey": "bigint",
        "s_phone": "string",
        "s_acctbal": "double",
        "s_comment": "string",
    },
    "partsupp": {
        "ps_partkey": "bigint",
        "ps_suppkey": "bigint",
        "ps_availqty": "int",
        "ps_supplycost": "double",
        "ps_comment": "string",
    },
    "nation": {
        "n_nationkey": "bigint",
        "n_name": "string",
        "n_regionkey": "bigint",
        "n_comment": "string",
    },
    "region": {
        "r_regionkey": "bigint",
        "r_name": "string",
        "r_comment": "string",
    },
}
