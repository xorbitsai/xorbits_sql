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

import numpy as np
import pandas as pd
import pytest
import xorbits.pandas as xpd

from .. import execute


@pytest.fixture
def prepare_data():
    rs = np.random.RandomState(123)
    df = pd.DataFrame(
        {
            "a": [f"t_{i}" for i in rs.randint(1000, size=100)],
            "b": rs.random(100),
            "c": rs.randint(100, size=100),
        }
    )
    yield df


@pytest.fixture
def prepare_join_data():
    rs = np.random.RandomState(123)
    df = pd.DataFrame(
        {
            "a": [f"t_{i}" for i in rs.randint(1000, size=100)],
            "b": rs.random(100),
        }
    )
    yield df


def test_project_and_filter(prepare_data):
    raw_df = prepare_data.copy()
    df = xpd.DataFrame(raw_df)
    sql = """
    select a, b / 2 AS b
    from t1
    where c > 50
    """

    expected = raw_df[raw_df["c"] > 50]
    expected["b"] /= 2
    del expected["c"]
    result = execute(sql, tables={"t1": df}).fetch()
    pd.testing.assert_frame_equal(result, expected)


def test_aggregation(prepare_data):
    raw_df = prepare_data.copy()
    df = xpd.DataFrame(raw_df)
    sql = """
    select c, SUM(a) AS SUM_A, AVG(b / 2) as AVG_B
    from t1
    group by c
    LIMIT 30
    """

    expected = raw_df.copy()
    expected["b"] /= 2
    expected = expected.groupby("c", as_index=False).agg({"a": "sum", "b": "mean"})
    expected.columns = ["c", "sum_a", "avg_b"]
    expected = expected.iloc[:30]
    result = execute(sql, tables={"t1": df}).fetch()
    pd.testing.assert_frame_equal(result, expected)


def test_join(prepare_data, prepare_join_data):
    raw_df = prepare_data.copy()
    join_df = prepare_join_data.copy()

    sql = """
    select t1.a, t2.b / 2 AS b
    from t1, t2
    where t1.a == t2.a
    """

    expected = raw_df.merge(join_df, on="a")[["a", "b_y"]]
    expected["b_y"] /= 2
    expected.columns = ["a", "b"]
    result = execute(
        sql, tables={"t1": xpd.DataFrame(raw_df), "t2": xpd.DataFrame(join_df)}
    ).fetch()
    pd.testing.assert_frame_equal(result, expected)


def test_sort(prepare_data):
    raw_df = prepare_data.copy()

    sql = """
    select a, b * 5 as b, c
    from t1
    order by c DESC
    limit 10
    """

    expected = raw_df.sort_values(by="c", ascending=False)
    expected["b"] *= 5
    # del expected['c']
    expected = expected.iloc[:10]
    result = execute(sql, tables={"t1": xpd.DataFrame(raw_df)}).fetch()
    pd.testing.assert_frame_equal(result, expected)
