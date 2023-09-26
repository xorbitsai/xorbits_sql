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

import itertools
import operator
from functools import lru_cache
from typing import Any, Callable

import pandas
import xorbits
import xorbits.numpy as np
import xorbits.pandas as pd
from sqlglot import MappingSchema, exp, planner
from xoscar.utils import TypeDispatcher, classproperty

from .errors import ExecuteError, UnsupportedError
from .table import Tables

_SQL_AGG_FUNC_TO_PD = {
    exp.Avg: "mean",
    exp.Count: "count",
    exp.Max: "max",
    exp.Min: "min",
    exp.Sum: "sum",
    exp.Stddev: "std",
    exp.Variance: "var",
}

_SQLGLOT_TYPE_TO_DTYPE = {
    "float": "float32",
    "double": "float64",
    "int": "int32",
    "tinyint": "int8",
    "smallint": "int16",
    "bigint": "int64",
}


class XorbitsExecutor:
    def __init__(
        self, tables: Tables | None = None, schema: MappingSchema | None = None
    ):
        self.tables = tables or Tables()
        self.schema = schema

    @classproperty
    @lru_cache(1)
    def _exp_visitors(cls) -> TypeDispatcher:
        dispatcher = TypeDispatcher()
        for func in exp.ALL_FUNCTIONS:
            dispatcher.register(func, cls._func)
        dispatcher.register(exp.Alias, cls._alias)
        dispatcher.register(exp.Binary, cls._func)
        dispatcher.register(exp.Boolean, cls._boolean)
        dispatcher.register(exp.Case, cls._case)
        dispatcher.register(exp.Cast, cls._cast)
        dispatcher.register(exp.Column, cls._column)
        dispatcher.register(exp.Extract, cls._extract)
        dispatcher.register(exp.In, cls._in)
        dispatcher.register(exp.Literal, cls._literal)
        dispatcher.register(exp.Null, cls._null)
        dispatcher.register(exp.Unary, cls._func)
        dispatcher.register(exp.Ordered, cls._ordered)
        dispatcher.register(exp.Var, cls._var)
        return dispatcher

    @classmethod
    def _visit_exp(
        cls, expr: exp.Expression, context: dict[str, pd.DataFrame]
    ) -> pd.DataFrame | pd.Series:
        try:
            visitor = cls._exp_visitors.get_handler(type(expr))
        except KeyError:
            raise UnsupportedError(
                f"Unsupported expression: {expr}, type: {type(expr)}"
            )
        return visitor(expr, context)

    @staticmethod
    def _literal(literal: exp.Literal, context: dict[str, pd.DataFrame]):
        if literal.is_string:
            return literal.this
        elif literal.is_int:
            return int(literal.this)
        elif literal.is_star:
            return slice(None)
        else:
            return float(literal.this)

    @staticmethod
    def _boolean(boolean: exp.Boolean, context: dict[str, pd.DataFrame]):
        return True if boolean.this else False

    @staticmethod
    def _null(null: exp.Null, context: dict[str, pd.DataFrame]):
        return None

    @staticmethod
    def _var(var: exp.Var, context: dict[str, pd.DataFrame]):
        return var.name

    @classmethod
    def _cast(
        cls,
        cast: exp.Cast,
        context: dict[str, pd.DataFrame],
    ):
        this = cls._visit_exp(cast.this, context)
        to = getattr(exp.DataType.Type, str(cast.to))

        if to == exp.DataType.Type.DATE:
            if pandas.api.types.is_scalar(this):
                return pandas.to_datetime(this).normalize()
            else:
                return pd.to_datetime(this).dt.normalize()
        elif to in (exp.DataType.Type.DATETIME, exp.DataType.Type.TIMESTAMP):
            return pd.to_datetime(this)
        elif to == exp.DataType.Type.BOOLEAN:
            if pandas.api.types.is_scalar(this):
                return bool(this)
            else:
                # TODO: convert to arrow string when it's default in pandas
                return this.astype(bool)
        elif to in exp.DataType.TEXT_TYPES:
            if pandas.api.types.is_scalar(this):
                return str(this)
            else:
                # TODO: convert to arrow string when it's default in pandas
                return this.astype(str)
        elif str(cast.to) in _SQLGLOT_TYPE_TO_DTYPE:
            pd_type = _SQLGLOT_TYPE_TO_DTYPE[str(cast.to)]
            if pandas.api.types.is_scalar(this):
                return pandas.Series([this], dtype=pd_type)[0]
            else:
                return this.astype(pd_type)
        else:
            raise NotImplementedError(f"Casting {cast.this} to '{to}' not implemented.")

    @staticmethod
    def _column(column: exp.Column, context: dict[str, pd.DataFrame]) -> pd.Series:
        return context[column.table][column.name]

    @classmethod
    def _alias(
        cls,
        alias: exp.Alias,
        context: dict[str, pd.DataFrame],
    ) -> pd.Series:
        return cls._visit_exp(alias.this, context).rename(alias.output_name)

    @classmethod
    def _extract(cls, extract: exp.Extract, context: dict[str, pd.DataFrame]):
        name = cls._visit_exp(extract.this, context)
        inp = cls._visit_exp(extract.expression, context)
        if name.upper() in ("YEAR", "MONTH", "DAY"):
            return getattr(inp.dt, name.lower()).astype("int64")
        else:
            raise NotImplementedError

    @classmethod
    def _in(cls, in_: exp.In, context: dict[str, pd.DataFrame]):
        this = cls._visit_exp(in_.this, context)
        expressions = {cls._visit_exp(e, context) for e in in_.args.get("expressions")}
        return this.isin(expressions)

    @classmethod
    def _case(cls, case: exp.Case, context: dict[str, pd.DataFrame]):
        this = cls._visit_exp(case.this, context) if case.this else None
        default = case.args.get("default")
        chain = (
            cls._visit_exp(default, context)
            if isinstance(default, exp.Expression)
            else default
        )

        for e in reversed(case.args["ifs"]):
            true = e.args.get("true")
            true = (
                cls._visit_exp(true, context)
                if isinstance(true, exp.Expression)
                else true
            )
            condition = cls._visit_exp(e.this, context)
            if this is not None:
                condition = this == condition
            chain = np.where(condition, true, chain)
            index = getattr(condition, "index", getattr(true, "index", None))
            assert index is not None
            chain = pd.Series(chain, index=index)

        return chain

    @classproperty
    @lru_cache(1)
    def _operator_visitors(cls) -> TypeDispatcher:
        dispatcher = TypeDispatcher()
        dispatcher.register(exp.Add, operator.add)
        dispatcher.register(exp.And, operator.and_)
        dispatcher.register(exp.EQ, operator.eq)
        dispatcher.register(exp.Div, operator.truediv)
        dispatcher.register(exp.GT, operator.gt)
        dispatcher.register(exp.GTE, operator.ge)
        dispatcher.register(exp.Is, cls._is)
        dispatcher.register(exp.LT, operator.lt)
        dispatcher.register(exp.LTE, operator.le)
        dispatcher.register(exp.Mul, operator.mul)
        dispatcher.register(exp.NEQ, operator.ne)
        dispatcher.register(exp.Not, operator.neg)
        dispatcher.register(exp.Or, operator.or_)
        dispatcher.register(exp.Paren, lambda x: x)
        dispatcher.register(exp.Like, cls._like)
        dispatcher.register(exp.Sub, operator.sub)
        return dispatcher

    @classmethod
    def _func(cls, func: exp.Expression, context: dict[str, pd.DataFrame]) -> pd.Series:
        values = [
            cls._visit_exp(arg, context) if isinstance(arg, exp.Expression) else arg
            for arg in func.args.values()
        ]
        try:
            func = cls._operator_visitors.get_handler(type(func))
        except KeyError:
            raise UnsupportedError(
                f"Unsupported expression: {func}, type: {type(func)}"
            )
        return func(*values)

    @classmethod
    def _like(cls, left: pd.Series, right: str):
        r = right.replace("_", ".").replace("%", ".*")
        return left.str.contains(r, regex=True)

    @classmethod
    def _is(cls, left: pd.Series, right: Any):
        if right is None:
            return left.isnull()
        else:
            return left == right

    def execute(self, plan: planner.Plan) -> pd.DataFrame:
        finished = set()
        queue = set(plan.leaves)
        context: dict[planner.Step, dict[str, pd.DataFrame]] = dict()

        while queue:
            node = queue.pop()
            try:
                current_context = {
                    name: df
                    for dep in node.dependencies
                    for name, df in context[dep].items()
                }

                if isinstance(node, planner.Scan):
                    context[node] = self.scan(node, current_context)
                elif isinstance(node, planner.Aggregate):
                    context[node] = self.aggregate(node, current_context)
                elif isinstance(node, planner.Join):
                    context[node] = self.join(node, current_context)
                elif isinstance(node, planner.Sort):
                    context[node] = self.sort(node, current_context)
                elif isinstance(node, planner.SetOperation):
                    context[node] = self.set_operation(node, current_context)
                else:
                    raise NotImplementedError

                finished.add(node)

                for dep in node.dependents:
                    if all(d in context for d in dep.dependencies):
                        queue.add(dep)

                for dep in node.dependencies:
                    if all(d in finished for d in dep.dependents):
                        context.pop(dep)
            except Exception as e:
                raise ExecuteError(f"Step '{node.id}', failed: {e}") from e

        root = plan.root
        df = context[root][root.name]
        xorbits.run(df)
        return df

    def scan(
        self, step: planner.Scan, context: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        source = step.source

        if source and isinstance(source, exp.Expression):
            source = source.name or source.alias

        if source is None:
            return dict()
        elif source in context:
            if not step.projections and not step.condition:
                return {step.name: context[source]}
            df = context[source]
        elif isinstance(step.source, exp.Table) and isinstance(
            step.source.this, exp.ReadCSV
        ):
            context = self._scan_csv(step)
            df = next(iter(context.values()))
        else:
            table = self.tables.find(step.source)
            df = table.df
            context = {step.source.alias_or_name: df}

        return {step.name: self._project_and_filter(step, context, df)}

    @staticmethod
    def _schema_to_dtype(schema: dict[str, str]) -> dict[str, str]:
        result = dict()
        for name, type_name in schema.items():
            try:
                result[name] = _SQLGLOT_TYPE_TO_DTYPE[type_name.lower()]
            except KeyError:
                continue
        return result

    def _scan_csv(self, step: planner.Scan) -> dict[str, pd.DataFrame]:
        alias = step.source.alias
        source: exp.ReadCSV = step.source.this

        args = source.expressions
        filename = source.name

        delimiter = ","
        args = iter(arg.name for arg in args)
        for k, v in zip(args, args):
            if k == "delimiter":
                delimiter = v

        dtype = None
        if self.schema and alias in self.schema.mapping:
            dtype = self._schema_to_dtype(self.schema.mapping[alias])

        df = pd.read_csv(filename, sep=delimiter, dtype=dtype)
        return {alias: df}

    def _project_and_filter(
        self, step: planner.Scan, context: dict[str, pd.DataFrame], df: pd.DataFrame
    ) -> pd.DataFrame:
        condition = self._visit_exp(step.condition, context) if step.condition else None
        if step.projections:
            out = dict()
            for projection in step.projections:
                out[projection.alias_or_name] = self._visit_exp(projection, context)
            df = pd.DataFrame(out)
        if condition is not None:
            df = df[condition]
        if isinstance(step.limit, int):
            df = df.iloc[: step.limit]
        return df

    def aggregate(
        self, step: planner.Aggregate, context: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        df = context[step.source].copy()
        group_by = [
            self._visit_exp(g, context).rename(f"__g_{i}")
            for i, g in enumerate(step.group.values())
        ]
        distinct_fields = set()

        if step.operands:
            for op in step.operands:
                if isinstance(op.this, exp.Star):
                    df[op.alias_or_name] = 1
                elif isinstance(op.this, exp.Distinct):
                    # process input of distinct
                    df[op.alias_or_name] = self._visit_exp(
                        op.this.expressions[0], context
                    )
                    distinct_fields.add(op.alias_or_name)
                else:
                    df[op.alias_or_name] = self._visit_exp(op.this, context)

        aggregations: dict[str, pd.NamedAgg] = dict()
        temp_field_id_gen = itertools.count(1)
        post_processes: list[Callable] = []
        for agg_alias in step.aggregations:
            self._process_agg_alias(
                agg_alias,
                aggregations,
                df,
                distinct_fields,
                context,
                temp_field_id_gen,
                post_processes,
            )

        names = list(step.group)
        if aggregations:
            names += list(aggregations)
            if step.group:
                result = df.groupby(group_by).agg(**aggregations).reset_index()
            else:
                df["__g_0"] = 1
                result = df.groupby("__g_0").agg(**aggregations).reset_index(drop=True)

            result.columns = names
        else:
            result = pd.DataFrame(dict(zip(names, group_by))).drop_duplicates()

        if post_processes:
            for p in post_processes:
                p(result)
            locs = [
                i
                for i, n in enumerate(result.dtypes.index)
                if not n.startswith("__op_") and not n.startswith("__agg_")
            ]
            result = result.iloc[:, locs]

        if aggregations and not step.group:

            def _ensure_data(data: pandas.DataFrame) -> pandas.DataFrame:
                if not len(data):
                    data = type(data)([[None] * len(data.dtypes)], columns=data.columns)
                return data

            result = result.rechunk({1: len(result.dtypes)}).map_chunk(_ensure_data)

        if step.projections or step.condition:
            result = self._project_and_filter(
                step, {step.name: result, **{name: result for name in context}}, result
            )

        if isinstance(step.limit, int):
            result = result.iloc[: step.limit]

        return {step.name: result}

    def _ensure_col(
        self, agg: exp.AggFunc, df: pd.DataFrame, context: dict[str, pd.DataFrame]
    ) -> str:
        col = agg.this
        col_name = col.alias_or_name
        if col_name not in df.dtypes:
            df[col_name] = self._visit_exp(col, context)
        return col_name

    @staticmethod
    def _get_agg_func(agg: exp.AggFunc, distinct_fields: set[str]):
        if distinct_fields and agg.this.alias_or_name in distinct_fields:
            return "nunique"
        else:
            return _SQL_AGG_FUNC_TO_PD[type(agg)]

    def _process_agg_alias(
        self,
        op: exp.Alias,
        aggregations: dict[str, pandas.NamedAgg],
        df: pd.DataFrame,
        distinct_fields: set[str],
        context: dict[str, pd.DataFrame],
        temp_field_id_gen: itertools.count,
        post_processes: list[Callable],
    ):
        agg = op.this
        out_name = op.alias_or_name

        try:
            aggfunc = self._get_agg_func(agg, distinct_fields)
        except KeyError:
            # not agg function, check if operator like unary or binary
            # if so, check args if any is an agg func
            try:
                self._process_agg_arg(
                    agg,
                    aggregations,
                    df,
                    distinct_fields,
                    context,
                    temp_field_id_gen,
                    out_name,
                    post_processes,
                )
            except KeyError:
                raise UnsupportedError(
                    f"Unsupported aggregate function: {agg}, type: {type(agg)}"
                )
        else:
            col = self._ensure_col(agg, df, context)
            aggregations[out_name] = pd.NamedAgg(column=col, aggfunc=aggfunc)

    def _process_agg_arg(
        self,
        op: exp.Unary | exp.Binary,
        aggregations: dict[str, pandas.NamedAgg],
        df: pd.DataFrame,
        distinct_fields: set[str],
        context: dict[str, pd.DataFrame],
        temp_field_id_gen: itertools.count,
        name: str,
        post_processes: list[Callable],
    ):
        func = self._operator_visitors.get_handler(type(op))
        args = []
        for arg in op.args.values():
            try:
                aggfunc = self._get_agg_func(arg, distinct_fields)
            except KeyError:
                if isinstance(arg, exp.Literal):
                    args.append(self._literal(arg, None))  # type: ignore
                else:
                    out_name = f"__op_{next(temp_field_id_gen)}"
                    self._process_agg_arg(
                        arg,
                        aggregations,
                        df,
                        distinct_fields,
                        context,
                        temp_field_id_gen,
                        out_name,
                        post_processes,
                    )
                    args.append(operator.itemgetter(out_name))
            else:
                out_name = f"__agg_{next(temp_field_id_gen)}"
                args.append(operator.itemgetter(out_name))
                col = self._ensure_col(arg, df, context)
                aggregations[out_name] = pd.NamedAgg(column=col, aggfunc=aggfunc)

        def post_process(aggregated: pd.DataFrame):
            new_args = []
            for arg in args:
                if callable(arg):
                    new_args.append(arg(aggregated))
                else:
                    new_args.append(arg)
            aggregated[name] = func(*new_args)

        post_processes.append(post_process)

    def join(
        self, step: planner.Join, context: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        source = step.name
        source_df = context[source]
        source_context = {source: source_df}
        column_slices = {source: slice(0, len(source_df.dtypes))}
        df = None

        for name, join in step.joins.items():
            df = context[name]
            join_context = {name: df}
            start = max(r.stop for r in column_slices.values())
            column_slices[name] = slice(start, len(df.dtypes) + start)

            if join.get("source_key"):
                df = self._hash_join(join, source_df, source_context, df, join_context)
            else:
                df = self._nested_loop_join(source_df, df)

            source_context = {
                name: df.iloc[:, column_slice]
                for name, column_slice in column_slices.items()
            }

            condition = self._visit_exp(join["condition"], source_context)
            if condition is not True:
                df = df.iloc[condition]
                source_context = {
                    name: df.iloc[:, column_slice]
                    for name, column_slice in column_slices.items()
                }

            source_df = df

        if not step.condition and not step.projections:
            return source_context

        sink = self._project_and_filter(step, source_context, df)

        if step.projections:
            return {step.name: sink}
        else:
            return {name: sink for name in source_context}

    def _nested_loop_join(
        self,
        source_df: pd.DataFrame,
        join_df: pd.DataFrame,
    ) -> pd.DataFrame:
        def func(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            if pandas.__version__ >= "1.2.0":
                result = left.merge(right, how="cross")
            else:
                left["_on"] = 1
                right["_on"] = 1
                result = left.merge(right, on="_on")
            result.columns = left.dtypes.index.tolist() + right.dtypes.index.tolist()
            return result

        return source_df.cartesian_chunk(join_df, func)

    def _hash_join(
        self,
        join: dict,
        source_df: pd.DataFrame,
        source_context: dict[str, pd.DataFrame],
        join_df: pd.DataFrame,
        join_context: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        cols = []

        source_df = source_df.copy()
        cols.extend(source_df.dtypes.index.tolist())
        left_ons = []
        for i, source_key in enumerate(join["source_key"]):
            col_name = f"__on_{i}"
            left_ons.append(col_name)
            source_df[col_name] = self._visit_exp(source_key, source_context)

        join_df = join_df.copy()
        cols.extend(join_df.dtypes.index.tolist())
        right_ons = []
        for i, join_key in enumerate(join["join_key"]):
            col_name = f"__on_{i}"
            right_ons.append(col_name)
            join_df[col_name] = self._visit_exp(join_key, join_context)

        how = "inner"
        if join.get("side") == "LEFT":
            how = "left"
        if join.get("side") == "RIGHT":
            how = "right"

        result = source_df.merge(join_df, how=how, left_on=left_ons, right_on=right_ons)
        ilocs = [
            i
            for i, col in enumerate(result.dtypes.index)
            if not col.startswith("__on_")
        ]
        result = result.iloc[:, ilocs]
        result.columns = cols
        return result

    @classmethod
    def _ordered(cls, ordered: exp.Ordered, context: dict[str, pd.DataFrame]):
        return (
            cls._visit_exp(ordered.this, context),
            True if ordered.args.get("desc") else False,
            "first" if ordered.args.get("nulls_first") else "last",
        )

    def sort(
        self, step: planner.Sort, context: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        df = context[step.name]
        df = df.copy()
        for projection in step.projections:
            df[projection.alias_or_name] = self._visit_exp(projection, context)

        sort_context = {"": df, **context}

        sort = [self._visit_exp(k, sort_context) for k in step.key]
        sort_cols = []
        ascendings = []
        na_position = None
        for i, (s, descending, cur_na_position) in enumerate(sort):
            sort_col = f"__s_{i}"
            sort_cols.append(sort_col)
            ascendings.append(not descending)
            if na_position is None:
                na_position = cur_na_position
            elif na_position != cur_na_position:
                raise NotImplementedError("nulls_first must be same for all sort keys")
            df[sort_col] = s

        df = df.sort_values(by=sort_cols, ascending=ascendings, na_position=na_position)
        df = df[[p.alias_or_name for p in step.projections]]

        if isinstance(step.limit, int):
            df = df.iloc[: step.limit]

        return {step.name: df.reset_index(drop=True)}

    def set_operation(
        self, step: planner.SetOperation, context: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError
