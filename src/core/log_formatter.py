"""
Colored terminal log formatter.
Attaches as a global logging handler so every logger.* call
across the whole application prints to the terminal with color.
"""
import logging
import sys


class ColoredTerminalFormatter(logging.Formatter):
    """Formatter that adds ANSI colors and icons based on log level."""

    _COLORS = {
        logging.DEBUG:    "\033[90m",    # gray
        logging.INFO:     "\033[96m",    # cyan
        logging.WARNING:  "\033[93m",    # yellow
        logging.ERROR:    "\033[91m",    # red
        logging.CRITICAL: "\033[91;1m",  # bold red
    }
    _ICONS = {
        logging.DEBUG:    "·",
        logging.INFO:     "→",
        logging.WARNING:  "⚠",
        logging.ERROR:    "✘",
        logging.CRITICAL: "✘",
    }
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        icon = self._ICONS.get(record.levelno, " ")
        reset = self._RESET

        timestamp = self.formatTime(record, "%H:%M:%S")
        name = record.name.split(".")[-1]  # short module name
        level = record.levelname

        msg = record.getMessage()
        base = f"{self._DIM}{timestamp}{reset} {color}{icon} {level:<8}{reset} {self._DIM}[{name}]{reset} {msg}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            base += f"\n{color}{record.exc_text}{reset}"

        return base


def setup_terminal_logging(level: int = logging.DEBUG) -> None:
    """
    Add a colored stderr handler to the root logger.
    Call once at startup — every logger in the app inherits it.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(ColoredTerminalFormatter())

    root = logging.getLogger()
    root.addHandler(handler)
