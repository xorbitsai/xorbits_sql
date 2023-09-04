<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits SQL: made pandas and SQL APIs work seamlessly together

[![PyPI Latest Release](https://img.shields.io/pypi/v/xorbits_sql.svg?style=for-the-badge)](https://pypi.org/project/xorbits_sql/)
[![License](https://img.shields.io/pypi/l/xorbits_sql.svg?style=for-the-badge)](https://github.com/xorbitsai/xorbits_sql/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/xorbits_sql/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/xorbits_sql/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

Xorbits SQL provides a SQL interface built on [Xorbits](https://github.com/xorbitsai/xorbits), 
allowing you to fluidly combine pandas and SQL to solve problems using the most familiar interface.
</div>
<br />

<div align="center">
<i><a href="https://join.slack.com/t/xorbitsio/shared_invite/zt-1z3zsm9ep-87yI9YZ_B79HLB2ccTq4WA">👉 Join our Slack community!</a></i>
</div>

**Xorbits SQL is still at an early development stage and under active improvement. 
Please feel free to provide feedback if you encounter any issues or have any suggestion.**

## Key features

🌟 **Pandas and SQL APIs in one**: The popular pandas and SQL APIs now work seamlessly together.

⚡️**Out-of-core and distributed capabilities**: Thanks to the underlying Xorbits execution engine, 
out-of-core and distributed runtimes are natively supported.

🔌 **Mainstream SQL dialects compatible**: By leveraging [SQLGlot](https://github.com/tobymao/sqlglot) as the SQL parser, 
Xorbits SQL is compatible with many dialects including DuckDB, Presto, Spark, Snowflake, and BigQuery.

## Getting Started
Xorbits SQL can be installed via pip from PyPI. It is highly recommended to create a new virtual
environment to avoid conflicts.

### Installation
```bash
$ pip install "xorbits_sql"
```

### Quick Start

Xorbits SQL provides a single API `execute()` which will return an Xorbits DataFrame.

```python
import xorbits.pandas as pd
import xorbits_sql as xsql

df = pd.DataFrame({"a": [1, 2, 3], "b": ['a', 'b', 'a']})
# SQL
sql = """
select b, AVG(a) as result
from t
group by b
"""
df2 = xsql.execute(
    sql,
    dialect=None,     # optional, replace with SQL dialect, e.g. "duckdb"
    tables={'t': df}  # register table, table name to Xorbits DataFrame
)
print(df2)
```

## License
[Apache 2](LICENSE)
