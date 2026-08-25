from __future__ import annotations
import enum
import typing
__all__: list[str] = ['alert', 'apply_all', 'async_logger', 'async_overflow_policy', 'basic_file_sink_mt', 'basic_file_sink_st', 'basic_logger_mt', 'basic_logger_st', 'color_mode', 'critical', 'daily_file_sink_mt', 'daily_file_sink_st', 'daily_logger_mt', 'daily_logger_st', 'debug', 'default_logger', 'drop', 'drop_all', 'error', 'fatal', 'flush_every', 'flush_on', 'get', 'get_level', 'info', 'level', 'logger', 'notice', 'null_sink_st', 'pattern_time_type', 'register_logger', 'rotating_file_sink_mt', 'rotating_file_sink_st', 'rotating_logger_mt', 'rotating_logger_st', 'set_default_logger', 'set_level', 'set_pattern', 'sink', 'stderr_color_mt', 'stderr_color_sink_mt', 'stderr_color_sink_st', 'stderr_color_st', 'stderr_logger_mt', 'stderr_logger_st', 'stderr_sink_mt', 'stderr_sink_st', 'stdout_color_mt', 'stdout_color_sink_mt', 'stdout_color_sink_st', 'stdout_color_st', 'stdout_logger_mt', 'stdout_logger_st', 'stdout_sink_mt', 'stdout_sink_st', 'trace', 'warn']
class _async_logger(logger):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class async_overflow_policy(enum.Enum):
    block: typing.ClassVar[async_overflow_policy]  # value = async_overflow_policy.block
    overrun_oldest: typing.ClassVar[async_overflow_policy]  # value = async_overflow_policy.overrun_oldest
class basic_file_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, truncate: bool = False) -> None:
        ...
    def filename(self) -> str:
        ...
class basic_file_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, truncate: bool = False) -> None:
        ...
    def filename(self) -> str:
        ...
class color_mode(enum.Enum):
    always: typing.ClassVar[color_mode]  # value = color_mode.always
    automatic: typing.ClassVar[color_mode]  # value = color_mode.automatic
    never: typing.ClassVar[color_mode]  # value = color_mode.never
class daily_file_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, hour: int = 0, minute: int = 0) -> None:
        ...
    def filename(self) -> str:
        ...
class daily_file_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, hour: int = 0, minute: int = 0) -> None:
        ...
    def filename(self) -> str:
        ...
class level(enum.Enum):
    alert: typing.ClassVar[level]  # value = level.alert
    critical: typing.ClassVar[level]  # value = level.critical
    debug: typing.ClassVar[level]  # value = level.debug
    err: typing.ClassVar[level]  # value = level.err
    fatal: typing.ClassVar[level]  # value = level.fatal
    info: typing.ClassVar[level]  # value = level.info
    notice: typing.ClassVar[level]  # value = level.notice
    off: typing.ClassVar[level]  # value = level.off
    trace: typing.ClassVar[level]  # value = level.trace
    warn: typing.ClassVar[level]  # value = level.warn
class logger:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, arg: str, sinks: list[sink]) -> None:
        """
        __init__(self, name: str, sink: pyspdlog.pyspdlog.sink) -> None
        __init__(self, name: str, sinks: collections.abc.Sequence[pyspdlog.pyspdlog.sink]) -> None
        """
    def alert(self, arg: str) -> None:
        ...
    def clone(self, arg: str) -> logger:
        ...
    def critical(self, arg: str) -> None:
        ...
    def debug(self, arg: str) -> None:
        ...
    def error(self, arg: str) -> None:
        ...
    def fatal(self, arg: str) -> None:
        ...
    def flush(self) -> None:
        ...
    def flush_on(self, arg: level) -> None:
        ...
    def info(self, arg: str) -> None:
        ...
    def level(self) -> level:
        ...
    def log(self, arg0: level, arg1: str) -> None:
        ...
    def name(self) -> str:
        ...
    def notice(self, arg: str) -> None:
        ...
    def set_level(self, arg: level) -> None:
        ...
    def set_pattern(self, pattern: str, time_type: pattern_time_type = ...) -> None:
        ...
    def should_log(self, arg: level) -> bool:
        ...
    def sinks(self) -> list[sink]:
        ...
    def trace(self, arg: str) -> None:
        ...
    def warn(self, arg: str) -> None:
        ...
class null_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        ...
class pattern_time_type(enum.Enum):
    local: typing.ClassVar[pattern_time_type]  # value = pattern_time_type.local
    utc: typing.ClassVar[pattern_time_type]  # value = pattern_time_type.utc
class rotating_file_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, max_size: int, max_files: int) -> None:
        ...
class rotating_file_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, filename: str, max_size: int, max_files: int) -> None:
        ...
class sink:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def level(self) -> level:
        ...
    def log(self, arg0: level, arg1: str) -> None:
        ...
    def set_level(self, arg: level) -> None:
        ...
    def set_pattern(self, arg: str) -> None:
        ...
class stderr_color_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        """
        __init__(self, mode: pyspdlog.pyspdlog.color_mode) -> None
        """
    def set_color(self, arg0: level, arg1: ...) -> None:
        ...
    def set_color_mode(self, arg: color_mode) -> None:
        ...
class stderr_color_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        """
        __init__(self, mode: pyspdlog.pyspdlog.color_mode) -> None
        """
    def set_color(self, arg0: level, arg1: ...) -> None:
        ...
    def set_color_mode(self, arg: color_mode) -> None:
        ...
class stderr_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        ...
class stderr_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        ...
class stdout_color_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        """
        __init__(self, mode: pyspdlog.pyspdlog.color_mode) -> None
        """
    def set_color(self, arg0: level, arg1: ...) -> None:
        ...
    def set_color_mode(self, arg: color_mode) -> None:
        ...
class stdout_color_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        """
        __init__(self, mode: pyspdlog.pyspdlog.color_mode) -> None
        """
    def set_color(self, arg0: level, arg1: ...) -> None:
        ...
    def set_color_mode(self, arg: color_mode) -> None:
        ...
class stdout_sink_mt(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        ...
class stdout_sink_st(sink):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self) -> None:
        ...
alert: nanobind.nb_func  # value = <nanobind.nb_func object>
apply_all: nanobind.nb_func  # value = <nanobind.nb_func object>
async_logger: nanobind.nb_func  # value = <nanobind.nb_func object>
basic_logger_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
basic_logger_st: nanobind.nb_func  # value = <nanobind.nb_func object>
critical: nanobind.nb_func  # value = <nanobind.nb_func object>
daily_logger_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
daily_logger_st: nanobind.nb_func  # value = <nanobind.nb_func object>
debug: nanobind.nb_func  # value = <nanobind.nb_func object>
default_logger: nanobind.nb_func  # value = <nanobind.nb_func object>
drop: nanobind.nb_func  # value = <nanobind.nb_func object>
drop_all: nanobind.nb_func  # value = <nanobind.nb_func object>
error: nanobind.nb_func  # value = <nanobind.nb_func object>
fatal: nanobind.nb_func  # value = <nanobind.nb_func object>
flush_every: nanobind.nb_func  # value = <nanobind.nb_func object>
flush_on: nanobind.nb_func  # value = <nanobind.nb_func object>
get: nanobind.nb_func  # value = <nanobind.nb_func object>
get_level: nanobind.nb_func  # value = <nanobind.nb_func object>
info: nanobind.nb_func  # value = <nanobind.nb_func object>
notice: nanobind.nb_func  # value = <nanobind.nb_func object>
register_logger: nanobind.nb_func  # value = <nanobind.nb_func object>
rotating_logger_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
rotating_logger_st: nanobind.nb_func  # value = <nanobind.nb_func object>
set_default_logger: nanobind.nb_func  # value = <nanobind.nb_func object>
set_level: nanobind.nb_func  # value = <nanobind.nb_func object>
set_pattern: nanobind.nb_func  # value = <nanobind.nb_func object>
stderr_color_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
stderr_color_st: nanobind.nb_func  # value = <nanobind.nb_func object>
stderr_logger_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
stderr_logger_st: nanobind.nb_func  # value = <nanobind.nb_func object>
stdout_color_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
stdout_color_st: nanobind.nb_func  # value = <nanobind.nb_func object>
stdout_logger_mt: nanobind.nb_func  # value = <nanobind.nb_func object>
stdout_logger_st: nanobind.nb_func  # value = <nanobind.nb_func object>
trace: nanobind.nb_func  # value = <nanobind.nb_func object>
warn: nanobind.nb_func  # value = <nanobind.nb_func object>
