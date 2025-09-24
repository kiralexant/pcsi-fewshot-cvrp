import logging
import sys
import time
from datetime import timedelta
from functools import wraps
from typing import Optional


def timecall(
    logger: logging.Logger, label: str | None = None, level: int = logging.INFO
):
    """
    Decorator factory: log wall time (timed by time.perf_counter) as a timedelta.
    Uses the provided logger; keeps function metadata via functools.wraps.
    """

    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (
                    time.perf_counter() - t0
                )  # high-res, monotonic. Only deltas are meaningful.
                logger.log(
                    level, "%s took %s", label or fn.__qualname__, timedelta(seconds=dt)
                )

        return wrapped

    return deco


class _FancyFormatter(logging.Formatter):
    # Simple ANSI colors if Rich isn't available
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "\x1b[38;5;244m",
        logging.INFO: "\x1b[38;5;39m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[41m\x1b[97m",
    }
    BASE_FMT = "▌ %(asctime)s │ %(name)s │ %(levelname)s │ %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        self._style._fmt = self.BASE_FMT
        self.datefmt = self.DATE_FMT
        return super().format(record)


class ScopedAdapter(logging.LoggerAdapter):
    def bind_scope(self, label: str = ""):
        self.extra["scope"] = label
        return self

    def clear_scope(self):
        self.extra.pop("scope", None)
        return self


def configure_logger(logger_name: Optional[str] = None, log_level: int = logging.INFO):
    # --- create a fancy logger just for this instance ---
    name = logger_name or f"FewShotCVRPLogger"
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:  # avoid duplicate handlers if re-instantiated
        handler: logging.Handler
        try:
            # Prefer Rich if installed (prettier output)
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(
                rich_tracebacks=False,
                show_path=False,
                show_time=True,
                markup=True,
            )
            fmt = logging.Formatter("%(message)s")  # Rich prints time/level nicely
            handler.setFormatter(fmt)
        except Exception:
            # Fallback: ANSI-colored StreamHandler
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(_FancyFormatter())

        logger.addHandler(handler)
        logger.propagate = False  # keep logs from duplicating to root

    return logger
