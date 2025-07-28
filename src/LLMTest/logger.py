import sys
from loguru import logger

def change_log_level(level="INFO"):
    logger.remove()

    log_format = (
        "<blue>[{time:YYYY-MM-DD HH:mm:ss,SSS}]</blue> "
        "<level>{level: <6}</level> "
        "{message}  "
        "<italic><cyan>({file.name}</cyan>:<cyan>{line}</cyan>:<cyan>{function})</cyan></italic>"
    )

    logger.add(sys.stderr, format=log_format, colorize=True, level=level)

change_log_level()

__all__ = ["logger"]
