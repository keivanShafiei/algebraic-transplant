"""Centralized logging configuration for the RBF-FD GNN Projection project."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    experiment_name: str = None,
) -> logging.Logger:
    """Configure logging with file and console handlers.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    log_level : int
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_to_file : bool
        Whether to write logs to file.
    log_to_console : bool
        Whether to print logs to console.
    experiment_name : str, optional
        Name for this experiment run. If None, uses timestamp.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get or create logger
    logger = logging.getLogger("rbffd_gnn")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{experiment_name}.log",
            mode="w",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging initialized: experiment={experiment_name}, level={logging.getLevelName(log_level)}")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger.

    Parameters
    ----------
    name : str, optional
        Submodule name (e.g., 'train', 'solver', 'projection').

    Returns
    -------
    logging.Logger
        Child logger with 'rbffd_gnn' as parent.
    """
    if name:
        return logging.getLogger(f"rbffd_gnn.{name}")
    return logging.getLogger("rbffd_gnn")
