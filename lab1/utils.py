import logging.config
from datetime import datetime
from logging import LoggerAdapter, getLogger
from os import path, makedirs
from typing import Dict, Any

from pythonjsonlogger.jsonlogger import JsonFormatter

UNSPECIFIED = 'unspecified'
__project_name__ = UNSPECIFIED
__version__ = '1.0.1'


def setup_logging(project_name: str):
    global __project_name__
    if __project_name__ != UNSPECIFIED:
        raise RuntimeError(f'Setup should be called only once!')
    __project_name__ = project_name

    log_directory = 'log'
    if not path.exists(log_directory):
        makedirs(log_directory)

    utc_time = datetime.utcnow().isoformat(timespec='seconds')
    log_filename = log_directory+f'/{utc_time}.log'.replace(':', '_')
    logging.config.fileConfig('logging.ini', defaults={'args': (log_filename, 'a', 'utf-8')})


class ExtraKeepLoggerAdapter(LoggerAdapter):
    """Keep logger instance's `extra` dictionary unchanged while writing logs with
    `extra` dictionary. This dictionary is used, in particular, for Splunk
    logger output.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Any:
        merged_context = self.extra.copy()

        if 'extra' in kwargs:
            merged_context.update(kwargs['extra'])

        kwargs['extra'] = merged_context
        return msg.format_map(merged_context), kwargs


def get_log(class_name: str) -> LoggerAdapter:
    if __project_name__ == UNSPECIFIED:
        raise RuntimeError(
            f'The project name ({__project_name__}) should be specified '
            f'using `setup_logging()` method of this module before log usage!'
        )
    _log = getLogger(class_name)

    extra = {
        'class_name': class_name,
        'project_name': __project_name__,
        'version': __version__,
    }
    return ExtraKeepLoggerAdapter(_log, extra)


def get_class_log(obj: Any) -> LoggerAdapter:
    return get_log(obj.__class__.__name__)

