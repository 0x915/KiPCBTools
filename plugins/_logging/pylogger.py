import copy
import datetime
import logging
import logging.handlers
import os
import sys

from typing import Callable, TextIO, Union


from .highlighter import default_highlight_str

from . import Color
from . import Logger

TOP = 100
FATAL = 75
ALERT = 70
CRITICAL = 65
ERROR = 55
WARN = 45
NOTICE = 36
INFO = 35
DEBUG = 25
TRACK = 15
NOTSET = 0

LEVEL_MAP = {
    TOP: "TOP",
    FATAL: "FATAL",
    ALERT: "ALERT",
    CRITICAL: "CRITICAL",
    ERROR: "ERROR",
    WARN: "WARN",
    NOTICE: "NOTICE",
    INFO: "INFO",
    DEBUG: "DEBUG",
    TRACK: "TRACK",
}

logging.addLevelName(TOP, "#")
logging.addLevelName(FATAL, "F")
logging.addLevelName(ALERT, "A")
logging.addLevelName(CRITICAL, "C")
logging.addLevelName(ERROR, "E")
logging.addLevelName(WARN, "W")
logging.addLevelName(NOTICE, "N")
logging.addLevelName(INFO, "I")
logging.addLevelName(DEBUG, "D")
logging.addLevelName(TRACK, "T")

if os.name == "nt":
    os.system("")  # enable ANSI escape processing in the Windows console


class Level:
    top = TOP
    fatal = FATAL
    alert = ALERT
    critical = CRITICAL
    error = ERROR
    err = ERROR
    warn = WARN
    notice = NOTICE
    info = INFO
    debug = DEBUG
    trace = TRACK


class MicrosecondFormatter(logging.Formatter):
    """Formatter that injects a `%(usecs)` field (6-digit microseconds) into records."""

    def __init__(
        self,
        pattern: str | None = None,
        date_format: str | None = None,
    ) -> None:
        super().__init__(pattern, date_format)
        self._uses_usecs: bool = "%(usecs)" in (pattern or "")

    def format(self, record: logging.LogRecord) -> str:
        if self._uses_usecs:
            record = copy.copy(record)
            record.usecs = int(round((record.created % 1) * 1_000_000))
        return super().format(record)


class ColorFormatter(MicrosecondFormatter):
    """Formatter that colors the timestamp tag gray and the rest per level."""

    def __init__(
        self,
        pattern: str | None = None,
        date_format: str | None = None,
        level_colors: dict[int, str] | None = None,
        default_color: str = Color.ctl.reset,
    ) -> None:
        super().__init__(pattern, date_format)
        self.level_colors: dict[int, str] = level_colors if level_colors is not None else {}
        self.default_color: str = default_color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        code = self.level_colors.get(record.levelno, self.default_color)
        if not code:
            return line
        idx = line.find(Color.ctl.reset)
        if idx != -1:
            end = idx + len(Color.ctl.reset)
            head = line[:end]
            tail = line[end:]
            lead = len(tail) - len(tail.lstrip(" "))
            return f"{head}{tail[:lead]}{code}{tail[lead:]}{Color.ctl.reset}"
        return f"{code}{line}{Color.ctl.reset}"


class PrefixLogger(Logger):
    Level = Level
    default_level = TRACK

    def __init__(
        self,
        obj: str | logging.Logger | None = None,
        level: int = default_level,
        sinks: list[logging.Handler] | None = None,
        pattern: str | None = None,
        dyer: Callable[[str], str] | None = None,
    ) -> None:
        self.__prefix: str = ""
        self.__disabled_msg_output: bool = False
        self.dyer_calls = dyer
        if sinks is None:
            sinks = [
                Config.ColorStdoutSink(pattern=pattern),
            ]
        if obj is None or isinstance(obj, str):
            name = f"{id(self):x}" if obj is None else obj
            self.__inst = logging.Logger(name)
            self.__inst.setLevel(level)
            self.__inst.propagate = False
            for sink in sinks:
                self.__inst.addHandler(sink)
        elif isinstance(obj, logging.Logger):
            self.__inst = obj
        else:
            raise TypeError(obj)
        return

    def __str__(self):
        prefix: str = self.prefix().strip(" <>")
        return f'<{self.__class__.__name__}:{id(self):x} inst={id(self.__inst):x} level={LEVEL_MAP.get(self.__inst.level, self.__inst.level)} prefix="{prefix}">'

    #

    def __get_obj_prefix(self, obj):
        if isinstance(obj, str):
            return f"<{obj}> "
        return f"<0x{id(obj):x}> "

    def level(self) -> int:
        return self.__inst.level

    def setLevel(self, v: int):
        self.__inst.setLevel(v)

    def prefix(self):
        return self.__prefix

    def setPrefix(self, v: str):
        if len(v) == 0:
            prefix: str = ""
            self.debug(f"Disable prefix.")
        else:
            prefix = self.__get_obj_prefix(v)
            self.debug(f"Update prefix to {v.strip()}.")
        self.__prefix = prefix

    def instance(self):
        return self.__inst

    def setInstance(self, v: logging.Logger):
        self.__inst = v

    def status(self):
        return self.__disabled_msg_output is False

    def disable(self):
        self.debug(f"disable msg output.")
        self.__disabled_msg_output = True
        return

    def enable(self):
        self.__disabled_msg_output = False
        self.debug(f"enable msg output.")
        return

    def objLogger(self, obj: str | object):
        if not isinstance(obj, str):
            name = f"0x{id(obj):x}"
        else:
            name = obj
        logger = PrefixLogger(obj=self.__inst, dyer=self.dyer_calls)
        logger.__prefix = logger.__get_obj_prefix(name)
        if not isinstance(obj, str):
            logger.debug(f"The prefix bound object {obj}.")
        return logger

    #

    def fatal(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(FATAL, f"{self.__prefix}{msg}")

    def alert(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(ALERT, f"{self.__prefix}{msg}")

    def critical(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(CRITICAL, f"{self.__prefix}{msg}")

    def error(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            ERROR,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def warn(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            WARN,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def notice(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            NOTICE,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def info(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            INFO,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def debug(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            DEBUG,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def trace(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(TRACK, f"{self.__prefix}{msg}")


class Config:
    # fmt: off
    default_log_level = NOTSET
    default_date_format: str     = "%y-%m-%d %H:%M:%S"
    default_stdout_pattern: str  = "\033[90m[%(asctime)s %(usecs)06d %(levelname)1s]\033[0m [%(name)s] %(message)s"
    default_file_pattern: str    = "[%(asctime)s %(usecs)06d %(levelname)1s] %(message)s"
    default_stream: TextIO       = sys.stdout
    default_stdout_level         = NOTSET
    default_file_level           = NOTSET

    default_fatal_color: str     = Color.fg.black    + Color.bg.magenta  + Color.ctl.bold
    default_alert_color: str     = Color.fg.white    + Color.bg.red      + Color.ctl.bold
    default_critical_color: str  = Color.fg.magenta                      + Color.ctl.bold
    default_err_color: str       = Color.fg.red
    default_warn_color: str      = Color.fg.yellow
    default_notice_color: str    = Color.fg.cyan
    default_info_color: str      = Color.fg.lgreen
    default_debug_color: str     = Color.fg.lblack
    default_trace_color: str     = Color.fg.lblue
    default_color: str     = Color.ctl.reset
    default_dyer: Callable[[str], str] = default_highlight_str
    # fmt: on

    ColorSinkT = Union[logging.StreamHandler, logging.FileHandler]

    @classmethod
    def level_colors(cls) -> dict[int, str]:
        return {
            TOP: cls.default_fatal_color,
            FATAL: cls.default_fatal_color,
            ALERT: cls.default_alert_color,
            CRITICAL: cls.default_critical_color,
            ERROR: cls.default_err_color,
            WARN: cls.default_warn_color,
            NOTICE: cls.default_notice_color,
            INFO: cls.default_info_color,
            DEBUG: cls.default_debug_color,
            TRACK: cls.default_trace_color,
            logging.CRITICAL: cls.default_critical_color,
            logging.ERROR: cls.default_err_color,
            logging.WARNING: cls.default_warn_color,
            logging.INFO: cls.default_info_color,
            logging.DEBUG: cls.default_debug_color,
            NOTSET: cls.default_color,
        }

    @staticmethod
    def SetSinkColorToDefault(sink: ColorSinkT):
        formatter = getattr(sink, "formatter", None)
        if isinstance(formatter, ColorFormatter):
            formatter.level_colors = Config.level_colors()

    @staticmethod
    def SetSinkColorToNone(sink: ColorSinkT):
        formatter = getattr(sink, "formatter", None)
        if isinstance(formatter, ColorFormatter):
            formatter.level_colors = {level: Color.ctl.reset for level in Config.level_colors()}

    @staticmethod
    def StdoutSink(
        level: int = default_stdout_level,
        pattern: None | str = None,
    ) -> logging.StreamHandler:
        sink = logging.StreamHandler(Config.default_stream)
        sink.setLevel(level)
        sink.setFormatter(
            MicrosecondFormatter(
                Config.default_stdout_pattern if pattern is None else pattern,
                Config.default_date_format,
            )
        )
        return sink

    @staticmethod
    def ColorStdoutSink(
        level: int = default_stdout_level,
        pattern: None | str = None,
    ) -> logging.StreamHandler:
        sink = logging.StreamHandler(Config.default_stream)
        sink.setLevel(level)
        sink.setFormatter(
            ColorFormatter(
                Config.default_stdout_pattern if pattern is None else pattern,
                Config.default_date_format,
                level_colors=Config.level_colors(),
            )
        )
        return sink

    @staticmethod
    def ColorFileSink(
        filename: str,
        mode: str = "a",
        level: int = default_file_level,
        pattern: None | str = None,
    ) -> logging.FileHandler:
        sink = logging.FileHandler(filename, mode=mode, encoding="utf-8")
        sink.setLevel(level)
        sink.setFormatter(
            ColorFormatter(
                Config.default_stdout_pattern if pattern is None else pattern,
                Config.default_date_format,
                level_colors=Config.level_colors(),
            )
        )
        return sink

    @staticmethod
    def DailyFileSink(
        filename: str,
        hour: int = 0,
        minute: int = 0,
        level: int = default_file_level,
        pattern: None | str = None,
        backup_count: int = 0,
    ) -> logging.handlers.TimedRotatingFileHandler:
        sink = logging.handlers.TimedRotatingFileHandler(
            filename,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
            atTime=datetime.time(hour, minute),
        )
        sink.setLevel(level)
        sink.setFormatter(
            MicrosecondFormatter(
                Config.default_file_pattern if pattern is None else pattern,
                Config.default_date_format,
            )
        )
        return sink

    @staticmethod
    def RotatingFileSink(
        filename: str,
        max_size: int,
        max_files: int,
        level: int = default_file_level,
        pattern: None | str = None,
    ) -> logging.handlers.RotatingFileHandler:
        sink = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=max_size,
            backupCount=max_files,
            encoding="utf-8",
        )
        sink.setLevel(level)
        sink.setFormatter(
            MicrosecondFormatter(
                Config.default_file_pattern if pattern is None else pattern,
                Config.default_date_format,
            )
        )
        return sink


logger = PrefixLogger(
    "default",
    PrefixLogger.Level.trace,
    [
        Config.ColorStdoutSink(),
        # Config.DailyFileSink("default.log"),
    ],
)


def test():
    msg = "~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?"
    print(f"\ntest = {msg}")

    print(f"\n{logger}.level = {logger.level()}")
    print(f"{logger}.handlers = {logger.instance().handlers}")
    logger.trace(msg)
    logger.debug(msg)
    logger.info(msg)
    logger.notice(msg)
    logger.warn(msg)
    logger.error(msg)
    logger.critical(msg)
    logger.alert(msg)
    logger.fatal(msg)

    logger.dyer_calls = Config.default_dyer
    print(f"\n{logger}.dyer = {logger.dyer_calls}")
    logger.trace(msg)
    logger.debug(msg)
    logger.info(msg)
    logger.notice(msg)
    logger.warn(msg)
    logger.error(msg)
    logger.critical(msg)
    logger.alert(msg)
    logger.fatal(msg)

    objlogger = logger.objLogger(logger)
    print(f"\n{objlogger}")
    objlogger.trace(msg)
    objlogger.debug(msg)
    objlogger.info(msg)
    objlogger.notice(msg)
    objlogger.warn(msg)
    objlogger.error(msg)
    objlogger.critical(msg)
    objlogger.alert(msg)
    objlogger.fatal(msg)

    objlogger.dyer_calls = None
    print(f"\n{objlogger}.dyer = {objlogger.dyer_calls}")
    objlogger.trace(msg)
    objlogger.debug(msg)
    objlogger.info(msg)
    objlogger.notice(msg)
    objlogger.warn(msg)
    objlogger.error(msg)
    objlogger.critical(msg)
    objlogger.alert(msg)
    objlogger.fatal(msg)

    print(f"\n{logger}.dyer = {logger.dyer_calls}")
    logger.trace(msg)
    logger.debug(msg)
    logger.info(msg)
    logger.notice(msg)
    logger.warn(msg)
    logger.error(msg)
    logger.critical(msg)
    logger.alert(msg)
    logger.fatal(msg)


if __name__ == "__main__":
    os.system("cls")
    test()
