import pandas as pd
import pytest
import importlib

from meval import compare_groups
from meval.config import settings
from meval.metrics import Count


COMPARE_GROUPS_MODULE = importlib.import_module("meval.compare_groups")


class _DummyProcess:
    name = "SpawnPoolWorker-1"


class _DummyMainProcess:
    name = "MainProcess"


def _df_minimal():
    return pd.DataFrame({"y_true": [True, False, True], "y_pred_prob": [0.8, 0.2, 0.9]})


def test_parallel_top_level_spawn_raises_early(monkeypatch):
    old_parallel = settings.parallel
    try:
        settings.update(parallel=True)

        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "current_process", lambda: _DummyMainProcess())
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "cpu_count", lambda: 8)
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "get_start_method", lambda allow_none=True: "spawn")
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "_is_main_module_top_level_python_file_call", lambda: True)

        def _pool_should_not_be_called(*args, **kwargs):
            raise AssertionError("Pool should not be created when early guard should fail first.")

        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "Pool", _pool_should_not_be_called)

        with pytest.raises(RuntimeError, match="top-level script code"):
            compare_groups(df=_df_minimal(), metrics=[Count()], group_by=None, min_subgroup_size=1)

    finally:
        settings.update(parallel=old_parallel)


def test_parallel_top_level_windows_default_spawn_raises_early(monkeypatch):
    old_parallel = settings.parallel
    try:
        settings.update(parallel=True)

        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "current_process", lambda: _DummyMainProcess())
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "cpu_count", lambda: 8)
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "get_start_method", lambda allow_none=True: None)
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "_is_main_module_top_level_python_file_call", lambda: True)
        monkeypatch.setattr(COMPARE_GROUPS_MODULE.sys, "platform", "win32")

        with pytest.raises(RuntimeError, match="top-level script code"):
            compare_groups(df=_df_minimal(), metrics=[Count()], group_by=None, min_subgroup_size=1)

    finally:
        settings.update(parallel=old_parallel)


def test_parallel_worker_reentry_raises(monkeypatch):
    old_parallel = settings.parallel
    try:
        settings.update(parallel=True)

        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "current_process", lambda: _DummyProcess())
        monkeypatch.setattr(COMPARE_GROUPS_MODULE, "cpu_count", lambda: 8)

        with pytest.raises(RuntimeError, match="non-main process"):
            compare_groups(df=_df_minimal(), metrics=[Count()], group_by=None, min_subgroup_size=1)

    finally:
        settings.update(parallel=old_parallel)
