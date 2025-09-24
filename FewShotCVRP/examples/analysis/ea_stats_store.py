import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

RunArray = Iterable[Tuple[int, float, float]]  # (evals, theta, best_so_far)


@dataclass
class EAStatsStore:
    _raw: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "run_id",
                "iter_idx",
                "evals",
                "theta",
                "best_so_far",
                "at_iteration_end",
                "cvrp_instance_name",
                "algorithm_name",
                "nn_trained_on_instance",
                "nn_training_method",
            ]
        )
    )
    _stats: Optional[pd.DataFrame] = None
    _next_run_id: int = 1  # sequential IDs start at 1

    # ------------------ ADD RUN ------------------
    def add_run(
        self,
        data: RunArray,
        cvrp_instance_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        nn_trained_on_instance: Optional[str] = None,
        nn_training_method: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        run_id: Optional[int] = None,  # allow manual override if needed
    ) -> int:
        arr = list(data)
        if not arr:
            raise ValueError("add_run: empty data")

        rid = int(run_id) if run_id is not None else self._next_run_id

        df = pd.DataFrame(arr, columns=["evals", "theta", "best_so_far"]).copy()

        # ---- enforce dtypes early ----
        df["evals"] = pd.to_numeric(df["evals"], errors="coerce").astype("int64")
        df["theta"] = pd.to_numeric(df["theta"], errors="coerce").astype("float64")
        df["best_so_far"] = pd.to_numeric(df["best_so_far"], errors="coerce").astype(
            "float64"
        )

        df = df.sort_values("evals").reset_index(drop=True)
        df["iter_idx"] = np.arange(1, len(df) + 1, dtype=np.int64)
        df["run_id"] = np.int64(rid)
        df["at_iteration_end"] = True
        df["cvrp_instance_name"] = cvrp_instance_name
        df["algorithm_name"] = algorithm_name
        df["nn_trained_on_instance"] = nn_trained_on_instance
        df["nn_training_method"] = nn_training_method

        if meta:
            for k, v in meta.items():
                df[k] = v

        base_cols = [
            "run_id",
            "iter_idx",
            "evals",
            "theta",
            "best_so_far",
            "at_iteration_end",
            "cvrp_instance_name",
            "algorithm_name",
            "nn_trained_on_instance",
            "nn_training_method",
        ]
        extra_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + extra_cols]

        # Avoid concat-deprecation warning on empty frames
        if self._raw.empty:
            self._raw = df.copy()
        else:
            self._raw = pd.concat([self._raw, df], ignore_index=True)

        self._stats = None
        self._next_run_id = max(self._next_run_id, rid + 1)
        return rid

    # ------------------ DELETE RUN ------------------
    def delete_run(
        self,
        cvrp_instance_name: str,
        algorithm_name: str,
        nn_training_method: Optional[str] = None,
        nn_trained_on_instance: Optional[str] = None,
        recompute_iter_idx: bool = True,
    ) -> dict:
        """
        Delete all rows whose (algorithm_name, nn_training_method, nn_trained_on_instance,
        cvrp_instance_name) match the provided values.
        Use None to match rows where the respective field is NA/None.

        Returns a summary dict with counts of removed rows and affected run_ids.
        """
        if self._raw.empty:
            return {"removed_rows": 0, "affected_runs": 0}

        df = self._raw

        def _match(colname: str, val):
            col = df[colname]
            # Treat None as "match NA"
            return col.isna() if val is None else (col == val)

        mask = (
            _match("algorithm_name", algorithm_name)
            & _match("cvrp_instance_name", cvrp_instance_name)
            & _match("nn_training_method", nn_training_method)
            & _match("nn_trained_on_instance", nn_trained_on_instance)
        )

        if not mask.any():
            return {"removed_rows": 0, "affected_runs": 0}

        affected_runs = set(df.loc[mask, "run_id"].tolist())
        removed_rows = int(mask.sum())

        # Keep everything not in the mask
        new_raw = df.loc[~mask].copy()

        # Sort and optionally re-number iter_idx per run
        new_raw = new_raw.sort_values(["run_id", "evals", "iter_idx"]).reset_index(
            drop=True
        )
        if recompute_iter_idx and not new_raw.empty:
            new_raw["iter_idx"] = new_raw.groupby("run_id").cumcount() + 1

        self._raw = new_raw
        self._stats = None  # invalidate cached aggregates

        # Note: we purposely do NOT change _next_run_id (so appended runs keep incrementing)
        return {"removed_rows": removed_rows, "affected_runs": len(affected_runs)}

    # ------------------ GROUPED STATS ------------------
    def compute_stats(
        self, *, group_by_nn_training: bool = True, group_by_nn_method: bool = True
    ) -> pd.DataFrame:
        """
        Compute stats per group of runs.

        Default grouping:
          (algorithm_name, cvrp_instance_name, nn_trained_on_instance, nn_training_method)

        If group_by_nn_training=False, drop nn_trained_on_instance from keys.
        If group_by_nn_method=False, drop nn_training_method from keys.

        Steps per group:
          - Build the union grid of evals (numeric, sorted).
          - Align each run to the grid via merge_asof(direction='backward').
          - Aggregate mean/std of best_so_far and theta at each eval.

        Returns a DataFrame with columns:
          ['algorithm_name','cvrp_instance_name',
           ('nn_trained_on_instance',) ('nn_training_method',)
           'evals','n_runs','best_so_far_mean','best_so_far_std','theta_mean','theta_std']
        """
        if self._raw.empty:
            cols = ["algorithm_name", "cvrp_instance_name"]
            if group_by_nn_training:
                cols.append("nn_trained_on_instance")
            if group_by_nn_method:
                cols.append("nn_training_method")
            cols += [
                "evals",
                "n_runs",
                "best_so_far_mean",
                "best_so_far_std",
                "theta_mean",
                "theta_std",
            ]
            self._stats = pd.DataFrame(columns=cols)
            return self._stats

        out_frames = []
        group_keys = ["algorithm_name", "cvrp_instance_name"]
        if group_by_nn_training:
            group_keys.append("nn_trained_on_instance")
        if group_by_nn_method:
            group_keys.append("nn_training_method")

        for key_vals, G in self._raw.groupby(group_keys, dropna=False):
            # coerce types inside the group
            gG = G.copy()
            gG["evals"] = pd.to_numeric(gG["evals"], errors="coerce").astype("int64")
            gG["theta"] = pd.to_numeric(gG["theta"], errors="coerce").astype("float64")
            gG["best_so_far"] = pd.to_numeric(
                gG["best_so_far"], errors="coerce"
            ).astype("float64")
            gG = gG.sort_values(["run_id", "evals"])

            grid = np.asarray(sorted(gG["evals"].unique()), dtype=np.int64)

            per_run = []
            for rid, g in gG.groupby("run_id", sort=False):
                g = (
                    g[["evals", "theta", "best_so_far"]]
                    .dropna(subset=["evals"])
                    .sort_values("evals")
                )
                gg = pd.DataFrame({"evals": grid})
                gg = pd.merge_asof(gg, g, on="evals", direction="backward")
                gg["run_id"] = np.int64(rid)
                per_run.append(gg)

            aligned = pd.concat(per_run, ignore_index=True)

            stats = aligned.groupby("evals", as_index=False).agg(
                n_runs=("best_so_far", lambda s: s.notna().sum()),
                best_so_far_mean=("best_so_far", "mean"),
                best_so_far_std=("best_so_far", "std"),
                theta_mean=("theta", "mean"),
                theta_std=("theta", "std"),
            )

            # prepend group columns in correct order
            # key_vals corresponds to group_keys in that order
            insert_pos = 0
            # We'll unpack flexibly, regardless of which flags are True
            kv_list = list(key_vals) if isinstance(key_vals, tuple) else [key_vals]

            # algorithm_name (index 0)
            algo = kv_list[0] if len(kv_list) > 0 else None
            # cvrp_instance_name (index 1)
            inst = kv_list[1] if len(kv_list) > 1 else None

            stats.insert(insert_pos, "algorithm_name", algo)
            insert_pos += 1
            stats.insert(insert_pos, "cvrp_instance_name", inst)
            insert_pos += 1

            # remaining optional keys come in order
            idx = 2
            if group_by_nn_training:
                trained = kv_list[idx] if len(kv_list) > idx else None
                stats.insert(insert_pos, "nn_trained_on_instance", trained)
                insert_pos += 1
                idx += 1
            if group_by_nn_method:
                method = kv_list[idx] if len(kv_list) > idx else None
                stats.insert(insert_pos, "nn_training_method", method)
                insert_pos += 1

            out_frames.append(stats)

        self._stats = (
            pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
        )
        return self._stats

    def get_raw_df(self) -> pd.DataFrame:
        return self._raw.copy()

    def get_stats_df(self) -> pd.DataFrame:
        return (
            self._stats.copy()
            if self._stats is not None
            else self.compute_stats().copy()
        )

    def get_stats_for(
        self,
        algorithm_name: str,
        cvrp_instance_name: str,
        nn_trained_on_instance: Optional[str] = None,
        nn_training_method: Optional[str] = None,
    ) -> pd.DataFrame:
        stats = self.get_stats_df()
        m = (stats["algorithm_name"] == algorithm_name) & (
            stats["cvrp_instance_name"] == cvrp_instance_name
        )
        if (
            nn_trained_on_instance is not None
            and "nn_trained_on_instance" in stats.columns
        ):
            m = m & (stats["nn_trained_on_instance"] == nn_trained_on_instance)
        if nn_training_method is not None and "nn_training_method" in stats.columns:
            m = m & (stats["nn_training_method"] == nn_training_method)
        return stats[m].reset_index(drop=True)

    # ------------------ SAVE ------------------
    def save_raw(
        self, path: str | Path, compression: str = "zstd", engine: str = "pyarrow"
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._raw.to_parquet(p, engine=engine, compression=compression, index=False)

    def save_stats(
        self, path: str | Path, compression: str = "zstd", engine: str = "pyarrow"
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.get_stats_df().to_parquet(
            p, engine=engine, compression=compression, index=False
        )

    # ------------------ LOAD & MERGE ------------------
    def load_raw(self, path: str | Path, engine: str = "pyarrow") -> None:
        """
        Load previously saved raw runs (file or partitioned dir) and merge.
        De-duplicates on (run_id, evals), keeping the latest row.
        Updates _next_run_id so future runs keep incrementing.
        """
        df_new = pd.read_parquet(path, engine=engine)

        # Ensure required columns exist (older files might not have the new ones)
        required = [
            "run_id",
            "iter_idx",
            "evals",
            "theta",
            "best_so_far",
            "at_iteration_end",
            "cvrp_instance_name",
            "algorithm_name",
            "nn_trained_on_instance",
            "nn_training_method",
        ]
        for col in required:
            if col not in df_new.columns:
                if col == "iter_idx" and {"evals", "run_id"}.issubset(df_new.columns):
                    df_new = df_new.sort_values(["run_id", "evals"])
                    df_new["iter_idx"] = (
                        df_new.groupby("run_id", sort=False).cumcount() + 1
                    )
                elif col == "at_iteration_end":
                    df_new["at_iteration_end"] = True
                else:
                    df_new[col] = pd.NA

        # Coerce core numeric types to avoid object dtypes creeping in
        for col, dtype in [
            ("evals", "int64"),
            ("theta", "float64"),
            ("best_so_far", "float64"),
            ("iter_idx", "int64"),
            ("run_id", "int64"),
        ]:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype(dtype)

        # Align schemas (preserve extra metadata columns)
        for col in df_new.columns:
            if col not in self._raw.columns:
                self._raw[col] = pd.NA
        for col in self._raw.columns:
            if col not in df_new.columns:
                df_new[col] = pd.NA

        merged = pd.concat([self._raw, df_new], ignore_index=True)
        merged = merged.sort_values(["run_id", "evals", "iter_idx"])
        merged = merged.drop_duplicates(subset=["run_id", "evals"], keep="last")

        # keep integer run_ids when possible
        merged["run_id"] = pd.to_numeric(merged["run_id"], errors="coerce").astype(
            "Int64"
        )

        self._raw = merged.reset_index(drop=True)
        self._stats = None

        # continue sequential IDs from max+1
        if self._raw["run_id"].notna().any():
            self._next_run_id = int(self._raw["run_id"].max()) + 1
        else:
            self._next_run_id = 1

    def load_stats(self, path: str | Path, engine: str = "pyarrow") -> None:
        stats = pd.read_parquet(path, engine=engine)
        # Allow both schemas (with and without the new columns)
        base_expected = {
            "algorithm_name",
            "cvrp_instance_name",
            "evals",
            "n_runs",
            "best_so_far_mean",
            "best_so_far_std",
            "theta_mean",
            "theta_std",
        }
        missing = base_expected.difference(stats.columns)
        if missing:
            raise ValueError(f"Stats file missing columns: {missing}")
        self._stats = stats.copy()

    # Optional: save each group's stats into separate files inside a folder
    def save_stats_by_group(
        self, dir_path: str | Path, compression: str = "zstd", engine: str = "pyarrow"
    ) -> None:
        df = self.get_stats_df()
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        def safe(s: Any) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s) if s is not None else "NA")

        for (algo, inst), g in df.groupby(
            ["algorithm_name", "cvrp_instance_name"], dropna=False
        ):
            fname = f"stats_algo={safe(algo)}__inst={safe(inst)}.parquet"
            g.to_parquet(p / fname, engine=engine, compression=compression, index=False)

    # ------------------ To update the old data ------------------
    def downsample_descent(
        self, lambda_: int, recompute_iter_idx: bool = False
    ) -> dict:
        """
        In-place: keep only rows with evals % lambda_ == 1 for the 'descent' algorithm,
        plus the final row of each descent run. Optionally re-number iter_idx per run.

        Returns a small summary dict with counts of removed/remaining rows for descent.
        """
        if self._raw.empty:
            return {"removed_rows": 0, "remaining_rows": 0, "desc_runs": 0}

        df = self._raw

        # Work only on descent algorithm rows
        mask_desc = df["algorithm_name"] == "descent"
        if not mask_desc.any():
            return {"removed_rows": 0, "remaining_rows": 0, "desc_runs": 0}

        sub = df.loc[mask_desc].copy()

        # Ensure numeric dtypes
        sub["evals"] = pd.to_numeric(sub["evals"], errors="coerce").astype("Int64")

        # Keep every lambda-th eval (shifted so 1, 1+λ, 1+2λ,...)
        # because descent typically logs each evaluation starting from 1
        keep_every_lambda = (sub["evals"] - 1) % lambda_ == 0

        # ...and always the final eval of each run
        max_eval_per_run = sub.groupby("run_id")["evals"].transform("max")
        keep_final = sub["evals"] == max_eval_per_run

        keep_mask = keep_every_lambda | keep_final
        removed = int((~keep_mask).sum())
        remain = int(keep_mask.sum())
        n_runs = sub["run_id"].nunique()

        # Rebuild _raw with filtered descent rows
        kept_desc = sub.loc[keep_mask]
        kept_desc = kept_desc.sort_values(["run_id", "evals", "iter_idx"])

        # Non-descent rows stay as-is
        others = df.loc[~mask_desc]

        new_raw = pd.concat([others, kept_desc], ignore_index=True)
        new_raw = new_raw.sort_values(["run_id", "evals", "iter_idx"])
        # Deduplicate any accidental duplicates on (run_id, evals)
        new_raw = new_raw.drop_duplicates(
            subset=["run_id", "evals"], keep="last"
        ).reset_index(drop=True)

        # Optionally re-number iteration indices within each run for cleanliness
        if recompute_iter_idx:
            new_raw = new_raw.sort_values(["run_id", "evals"]).reset_index(drop=True)
            new_raw["iter_idx"] = new_raw.groupby("run_id").cumcount() + 1

        self._raw = new_raw
        self._stats = None  # invalidate cached stats

        return {"removed_rows": removed, "remaining_rows": remain, "desc_runs": n_runs}

    def annotate_theta_training(self, trained_on_instance: str) -> int:
        """
        Retro-annotate: fill nn_trained_on_instance where algorithm_name == 'theta_control'
        and the field is null. Returns number of rows updated.
        """
        if self._raw.empty:
            return 0
        m = (self._raw["algorithm_name"] == "theta_control") & (
            self._raw["nn_trained_on_instance"].isna()
        )
        n = int(m.sum())
        if n:
            self._raw.loc[m, "nn_trained_on_instance"] = trained_on_instance
            self._stats = None
        return n

    def annotate_theta_training_method(self, training_method: str) -> int:
        """
        Retro-annotate: fill nn_training_method where algorithm_name == 'theta_control'
        and the field is null. Returns number of rows updated.
        """
        if self._raw.empty:
            return 0
        m = (self._raw["algorithm_name"] == "theta_control") & (
            self._raw["nn_training_method"].isna()
        )
        n = int(m.sum())
        if n:
            self._raw.loc[m, "nn_training_method"] = training_method
            self._stats = None
        return n

    # ------------------ SUMMARY ------------------
    def summarize(
        self,
        print_output: bool = True,
        return_per_run: bool = False,
    ):
        """
        Show a summary of the database grouped by
        (algorithm_name, cvrp_instance_name, nn_training_method, nn_trained_on_instance).

        Prints (by default) a table with:
        n_runs, total_rows, rows_per_run_mean, rows_per_run_std, rows_per_run_min, rows_per_run_max.

        Returns:
        summary_df  (and per_run_df if return_per_run=True)
        """
        if self._raw.empty:
            if print_output:
                print("Database is empty.")
            empty_cols = [
                "algorithm_name",
                "cvrp_instance_name",
                "nn_training_method",
                "nn_trained_on_instance",
                "n_runs",
                "total_rows",
                "rows_per_run_mean",
                "rows_per_run_std",
                "rows_per_run_min",
                "rows_per_run_max",
            ]
            empty_df = pd.DataFrame(columns=empty_cols)
            return (empty_df, pd.DataFrame(columns=[])) if return_per_run else empty_df

        # Count rows per run for each tuple
        keys = [
            "algorithm_name",
            "cvrp_instance_name",
            "nn_training_method",
            "nn_trained_on_instance",
            "run_id",
        ]
        per_run = (
            self._raw.groupby(keys, dropna=False)
            .size()
            .reset_index(name="rows_per_run")
            .sort_values(keys)
            .reset_index(drop=True)
        )

        # Aggregate to tuple-level summary
        tuple_keys = [
            "algorithm_name",
            "cvrp_instance_name",
            "nn_training_method",
            "nn_trained_on_instance",
        ]
        summary = (
            per_run.groupby(tuple_keys, dropna=False)
            .agg(
                n_runs=("run_id", "nunique"),
                total_rows=("rows_per_run", "sum"),
                rows_per_run_mean=("rows_per_run", "mean"),
                rows_per_run_std=("rows_per_run", "std"),
                rows_per_run_min=("rows_per_run", "min"),
                rows_per_run_max=("rows_per_run", "max"),
            )
            .reset_index()
            .sort_values(tuple_keys)
            .reset_index(drop=True)
        )

        # Pretty print
        if print_output:
            print(
                "\n=== Summary by (algorithm_name, cvrp_instance_name, nn_training_method, nn_trained_on_instance) ==="
            )
            # Format floats nicely; avoid scientific for large totals
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                160,
            ):
                print(summary.to_string(index=False))

            # If desired, also show per-run counts
            # (handy when you want to spot unbalanced runs within a tuple)
            # Uncomment if you want it always:
            # print("\n--- Per-run row counts ---")
            # with pd.option_context("display.max_rows", None, "display.width", 160):
            #     print(per_run.to_string(index=False))

        return (summary, per_run) if return_per_run else summary
