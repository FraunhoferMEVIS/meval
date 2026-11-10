"""Static image export that survives a stalled browser.

Plotly renders static images through Kaleido, which drives a headless Chromium.
Chromium occasionally stops answering while Kaleido waits for it without a time
limit, which would block the calling program forever. Exports made through this
module run with a deadline instead. When it expires, the browser started for that
export is killed, which makes Kaleido fail rather than wait, and the export is
retried.
"""

from collections.abc import Callable
from concurrent.futures import Future, wait
from functools import partial
from threading import Lock, Thread
from typing import TypeVar
from warnings import warn

import plotly.graph_objects as go
import psutil

T = TypeVar("T")

EXPORT_TIMEOUT_S = 120.0  # per attempt; longer than Kaleido's own 90 s render limit
MAX_EXPORT_ATTEMPTS = 3

_KILL_GRACE_S = 30.0  # how long a killed export gets to fail by itself
_BROWSER_NAME_PARTS = ("chrom", "edge", "brave", "vivaldi")  # browsers Kaleido can launch
_export_lock = Lock()


def _running_browsers() -> dict[int, psutil.Process]:
    """Return the browser processes descending from this process, keyed by PID."""
    browsers = {}
    for process in psutil.Process().children(recursive=True):
        try:
            if any(part in process.name().lower() for part in _BROWSER_NAME_PARTS):
                browsers[process.pid] = process
        except psutil.Error:  # exited in the meantime
            continue
    return browsers


def _kill_browsers_except(known_pids: set[int]) -> None:
    """Kill the browser processes descending from this process that are not known."""
    for pid, browser in _running_browsers().items():
        if pid not in known_pids:
            try:
                browser.kill()
            except psutil.Error:  # exited in the meantime
                continue


def _attempt_export(export: Callable[[], T]) -> T:
    """Run ``export`` once, killing the browser it started if the deadline passes."""
    known_pids = set(_running_browsers())
    result: Future[T] = Future()

    def run() -> None:
        try:
            result.set_result(export())
        except BaseException as error:
            result.set_exception(error)

    # A daemon thread, so that a stuck export cannot keep the interpreter alive.
    Thread(target=run, daemon=True).start()

    if not wait([result], timeout=EXPORT_TIMEOUT_S).done:
        _kill_browsers_except(known_pids)
        wait([result], timeout=_KILL_GRACE_S)
        raise TimeoutError(f"Image export did not finish within {EXPORT_TIMEOUT_S:g} s.")

    return result.result()


def export_with_retries(export: Callable[[], T]) -> T:
    """Run a plotly image export, retrying if it hangs or the browser fails.

    Parameters
    ----------
    export : Callable[[], T]
        Performs the export.

    Returns
    -------
    T
        The return value of ``export``.

    Raises
    ------
    TimeoutError
        If the last attempt did not finish within ``EXPORT_TIMEOUT_S``.
    Exception
        Whatever the last attempt raised. ``TypeError`` and ``ValueError`` indicate
        invalid arguments and are raised immediately, without retrying.
    """
    with _export_lock:  # browsers from concurrent exports could not be told apart
        for attempt in range(1, MAX_EXPORT_ATTEMPTS):
            try:
                return _attempt_export(export)
            except (TypeError, ValueError):
                raise
            except Exception as error:
                warn(
                    f"Image export attempt {attempt}/{MAX_EXPORT_ATTEMPTS} failed "
                    f"({error!r}); retrying.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return _attempt_export(export)


class ExportSafeFigure(go.Figure):
    """A plotly figure whose static image export cannot hang on a stalled browser."""

    def write_image(self, *args, **kwargs):
        """Write a static image; see ``plotly.graph_objects.Figure.write_image``."""
        return export_with_retries(partial(super().write_image, *args, **kwargs))

    def to_image(self, *args, **kwargs):
        """Return a static image; see ``plotly.graph_objects.Figure.to_image``."""
        return export_with_retries(partial(super().to_image, *args, **kwargs))
