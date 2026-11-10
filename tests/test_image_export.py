import threading
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

from meval import _image_export
from meval._image_export import ExportSafeFigure, export_with_retries
from meval.diags import pr_diag, prg_diag, rel_diag, roc_diag


class FakeBrowser:
    def __init__(self, on_kill: Callable[[], None] = lambda: None) -> None:
        self.kills = 0
        self._on_kill = on_kill

    def kill(self) -> None:
        self.kills += 1
        self._on_kill()


class ScriptedExport:
    """Runs one step per call, and repeats the last step once the script is exhausted."""

    def __init__(self, *steps: Callable[[], str]) -> None:
        self.steps = steps
        self.calls = 0

    def __call__(self, *args, **kwargs) -> str:
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return step()


@pytest.fixture()
def browsers(monkeypatch: pytest.MonkeyPatch) -> dict[int, FakeBrowser]:
    running: dict[int, FakeBrowser] = {}
    monkeypatch.setattr(_image_export, "_running_browsers", lambda: dict(running))
    monkeypatch.setattr(_image_export, "EXPORT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(_image_export, "_KILL_GRACE_S", 1.0)
    return running


def succeeds() -> str:
    return "image"


def browser_fails() -> str:
    raise RuntimeError("browser failed")


def stalls_until_killed(browsers: dict[int, FakeBrowser], pid: int) -> Callable[[], str]:
    """Behave like Kaleido: start a browser, then wait until it dies."""

    def step() -> str:
        killed = threading.Event()
        browsers[pid] = FakeBrowser(on_kill=killed.set)
        killed.wait(timeout=5)
        raise RuntimeError("browser closed")

    return step


def test_successful_export_runs_once(browsers: dict[int, FakeBrowser]) -> None:
    export = ScriptedExport(succeeds)

    assert export_with_retries(export) == "image"
    assert export.calls == 1


def test_stalled_export_kills_its_browser_and_retries(browsers: dict[int, FakeBrowser]) -> None:
    export = ScriptedExport(stalls_until_killed(browsers, pid=1), succeeds)

    with pytest.warns(RuntimeWarning, match=r"attempt 1/3 failed \(TimeoutError"):
        assert export_with_retries(export) == "image"

    assert export.calls == 2
    assert browsers[1].kills == 1


def test_gives_up_after_max_attempts(browsers: dict[int, FakeBrowser]) -> None:
    export = ScriptedExport(stalls_until_killed(browsers, pid=1))

    with pytest.warns(RuntimeWarning), pytest.raises(TimeoutError, match="did not finish"):
        export_with_retries(export)

    assert export.calls == _image_export.MAX_EXPORT_ATTEMPTS


def test_browser_started_before_the_export_survives(browsers: dict[int, FakeBrowser]) -> None:
    browsers[1] = FakeBrowser()
    export = ScriptedExport(stalls_until_killed(browsers, pid=2), succeeds)

    with pytest.warns(RuntimeWarning):
        export_with_retries(export)

    assert browsers[1].kills == 0
    assert browsers[2].kills == 1


def test_export_that_ignores_the_kill_still_returns_control(
    browsers: dict[int, FakeBrowser], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_image_export, "MAX_EXPORT_ATTEMPTS", 1)
    monkeypatch.setattr(_image_export, "_KILL_GRACE_S", 0.05)
    release = threading.Event()

    def never_returns() -> str:
        release.wait(timeout=5)
        return "too late"

    try:
        with pytest.raises(TimeoutError, match="did not finish"):
            export_with_retries(never_returns)
    finally:
        release.set()


def test_browser_failure_is_retried(browsers: dict[int, FakeBrowser]) -> None:
    export = ScriptedExport(browser_fails, succeeds)

    with pytest.warns(RuntimeWarning, match="browser failed"):
        assert export_with_retries(export) == "image"

    assert export.calls == 2


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_invalid_arguments_are_not_retried(browsers: dict[int, FakeBrowser], error: type[Exception]) -> None:
    def invalid() -> str:
        raise error("invalid")

    export = ScriptedExport(invalid)

    with pytest.raises(error):
        export_with_retries(export)

    assert export.calls == 1


@pytest.mark.parametrize("method", ["write_image", "to_image"])
def test_figure_export_methods_are_guarded(
    browsers: dict[int, FakeBrowser], monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    export = ScriptedExport(browser_fails, succeeds)
    monkeypatch.setattr(go.Figure, method, export, raising=True)

    with pytest.warns(RuntimeWarning):
        assert getattr(ExportSafeFigure(), method)("figure.png", scale=2) == "image"

    assert export.calls == 2


@pytest.mark.parametrize(
    "plot",
    [
        lambda df: rel_diag(df, add_risk_density=True),
        lambda df: rel_diag(df, add_risk_density=False),
        roc_diag,
        pr_diag,
        prg_diag,
    ],
)
def test_diagnostics_return_export_safe_figures(plot: Callable) -> None:
    df = pd.DataFrame(
        {
            "y_true": [False, False, True, True, False, True, False, True],
            "y_pred": [0.10, 0.25, 0.70, 0.90, 0.20, 0.80, 0.35, 0.65],
            "y_pred_prob": [0.10, 0.25, 0.70, 0.90, 0.20, 0.80, 0.35, 0.65],
        }
    )

    fig, _, _ = plot(df)

    assert isinstance(fig, ExportSafeFigure)
