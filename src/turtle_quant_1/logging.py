"""The centralised logging module for the project."""

import logging

from turtle_quant_1 import config

_LOG_LEVEL = logging.DEBUG if config.ENV == "d" else logging.INFO
# Uncomment the following line to override the default log level.
_LOG_LEVEL = logging.INFO

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

_root_logger = logging.getLogger("turtle_quant_1")
_root_logger.setLevel(_LOG_LEVEL)
_root_logger.addHandler(_handler)
_root_logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``turtle_quant_1``.

    Args:
        name: Typically the calling module's ``__name__``. If provided, the
            returned logger is named ``turtle_quant_1.<suffix>`` where
            ``<suffix>`` is ``name`` with the leading ``turtle_quant_1.``
            stripped (so passing ``__name__`` directly always works). If
            ``None``, the root ``turtle_quant_1`` logger is returned.
    """
    if name is None:
        return _root_logger
    suffix = name.removeprefix("turtle_quant_1.")
    return logging.getLogger(f"turtle_quant_1.{suffix}")
