import os

from typing import Callable, Union


from .highlighter import default_highlight_str

from . import pyspdlog as spdlog

from . import Color
from . import Logger


class PrefixLogger(Logger):
    Level = spdlog.level
    default_level = Level.trace

    def __init__(
        self,
        obj: str | spdlog.logger | None = None,
        level: Level = Level.trace,
        sinks: list[spdlog.sink] | None = None,
        pattern: str | None = None,
        dyer: Callable[[str], str] | None = None,
    ) -> None:
        self.__prefix: str = " "
        if sinks is None:
            sinks = [
                Config.ColorStdoutSink(pattern=pattern),
            ]
        self.__inst: spdlog.logger
        if obj is None:
            name = f"{id(self):x}"
            self.__inst = spdlog.logger(name, sinks)
            self.__inst.set_level(level)
        elif isinstance(obj, str):
            name = obj
            self.__inst = spdlog.logger(name, sinks)
            self.__inst.set_level(level)
        elif isinstance(obj, spdlog.logger):
            self.__inst = obj
        else:
            raise TypeError(obj)

        self.__disabled_msg_output: bool = False
        self.dyer_calls = dyer
        self.__inst.flush_on(self.Level.trace)
        return

    def __str__(self):
        prefix: str = self.prefix()
        if len(prefix) != 0:
            prefix = prefix[2:-2]
        else:
            prefix = ""
        return f'<{self.__class__.__name__}:{id(self):x} inst={id(self.__inst):x} level={self.__inst.level()} prefix="{prefix}">'

    #

    def __get_obj_prefix(self, obj):
        if isinstance(obj, str):
            return f" <{obj}> "
        return f" <0x{id(obj):x}> "

    def level(self) -> spdlog.level:
        return self.__inst.level()

    def setLevel(self, v: Level):
        self.__inst.set_level(v)

    def prefix(self):
        return self.__prefix

    def setPrefix(self, v: str):
        if len(v) == 0:
            prefix: str = " "
            self.debug(f"Disable prefix.")
        else:
            prefix = self.__get_obj_prefix(v)
            self.debug(f"Update prefix to {v.strip()}.")
        self.__prefix = prefix

    def instance(self):
        return self.__inst

    def setInstance(self, v: spdlog.logger):
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
        self.__inst.log(
            self.Level.fatal,
            f"{self.__prefix}{msg}",
        )

    def alert(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.alert,
            f"{self.__prefix}{msg}",
        )

    def critical(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.critical,
            f"{self.__prefix}{msg}",
        )

    def error(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.err,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def warn(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.warn,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def notice(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.notice,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def info(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.info,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def debug(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.debug,
            f"{self.__prefix}{'' if self.dyer_calls is None else Color.ctl.reset}{msg if self.dyer_calls is None else self.dyer_calls(msg)}",
        )

    def trace(self, msg: str):
        if self.__disabled_msg_output:
            return
        self.__inst.log(
            self.Level.trace,
            f"{self.__prefix}{msg}",
        )


class Config:
    # fmt: off
    default_stdout_pattern: str  = "\033[90m[%C-%m-%d %H:%M:%S %f %L]\033[0m %^[%n]%v%$"
    default_file_pattern: str    = "[%C-%m-%d %H:%M:%S %f %L] %v%$"
    default_stdout_level         = spdlog.level.trace
    default_file_level           = spdlog.level.trace
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

    ColorSinkT = Union[spdlog.stdout_color_sink_mt]

    @staticmethod
    def SetSinkColorToDefault(sink: ColorSinkT):
        sink.set_color(PrefixLogger.Level.fatal, Config.default_fatal_color)
        sink.set_color(PrefixLogger.Level.alert, Config.default_alert_color)
        sink.set_color(PrefixLogger.Level.critical, Config.default_critical_color)
        sink.set_color(PrefixLogger.Level.err, Config.default_err_color)
        sink.set_color(PrefixLogger.Level.warn, Config.default_warn_color)
        sink.set_color(PrefixLogger.Level.notice, Config.default_notice_color)
        sink.set_color(PrefixLogger.Level.info, Config.default_info_color)
        sink.set_color(PrefixLogger.Level.debug, Config.default_debug_color)
        sink.set_color(PrefixLogger.Level.trace, Config.default_trace_color)

    @staticmethod
    def SetSinkColorToNone(sink: ColorSinkT):
        sink.set_color(PrefixLogger.Level.fatal, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.alert, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.critical, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.err, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.warn, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.notice, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.info, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.debug, Color.ctl.reset)
        sink.set_color(PrefixLogger.Level.trace, Color.ctl.reset)

    @staticmethod
    def StdoutSink(
        level: PrefixLogger.Level = default_stdout_level,
        pattern: None | str = None,
    ) -> spdlog.stdout_sink_mt:
        sink = spdlog.stdout_sink_mt()
        sink.set_level(level)
        sink.set_pattern(Config.default_stdout_pattern if pattern is None else pattern)
        return sink

    @staticmethod
    def ColorStdoutSink(
        level: PrefixLogger.Level = default_stdout_level,
        pattern: None | str = None,
    ) -> spdlog.stdout_color_sink_mt:
        sink = spdlog.stdout_color_sink_mt()
        sink.set_level(level)
        sink.set_pattern(Config.default_stdout_pattern if pattern is None else pattern)
        sink.set_color(PrefixLogger.Level.fatal, Config.default_fatal_color)
        sink.set_color(PrefixLogger.Level.alert, Config.default_alert_color)
        sink.set_color(PrefixLogger.Level.critical, Config.default_critical_color)
        sink.set_color(PrefixLogger.Level.err, Config.default_err_color)
        sink.set_color(PrefixLogger.Level.warn, Config.default_warn_color)
        sink.set_color(PrefixLogger.Level.notice, Config.default_notice_color)
        sink.set_color(PrefixLogger.Level.info, Config.default_info_color)
        sink.set_color(PrefixLogger.Level.debug, Config.default_debug_color)
        sink.set_color(PrefixLogger.Level.trace, Config.default_trace_color)
        return sink

    @staticmethod
    def ColorFileSink(
        filename: str,
        mode: str = "a",
        level: PrefixLogger.Level = default_file_level,
        pattern: None | str = None,
    ) -> spdlog.basic_file_sink_mt:
        sink = spdlog.basic_file_sink_mt(filename, mode != "a")
        sink.set_level(level)
        sink.set_pattern(Config.default_stdout_pattern if pattern is None else pattern)
        return sink

    @staticmethod
    def DailyFileSink(
        filename: str,
        hour: int = 0,
        minute: int = 0,
        level: PrefixLogger.Level = default_file_level,
        pattern: None | str = None,
    ) -> spdlog.daily_file_sink_mt:
        sink = spdlog.daily_file_sink_mt(filename, hour, minute)
        sink.set_level(level)
        sink.set_pattern(Config.default_file_pattern if pattern is None else pattern)
        return sink


logger = PrefixLogger(
    "default",
    PrefixLogger.Level.trace,
    [
        Config.ColorStdoutSink(),
        # Config.DailyFileSink("default.log"),
    ],
)

spdlog.set_level(Config.default_stdout_level)
spdlog.set_pattern("\033[90m[%C-%m-%d %H:%M:%S %f %L]\033[0m %^%v%$")


def test():
    msg = "~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?"
    print(f"\ntest = {msg}")

    print(f"\n{spdlog}.level = {spdlog.get_level}")
    spdlog.trace(msg)
    spdlog.debug(msg)
    spdlog.info(msg)
    spdlog.notice(msg)
    spdlog.warn(msg)
    spdlog.error(msg)
    spdlog.critical(msg)
    spdlog.alert(msg)
    spdlog.fatal(msg)

    logger.dyer_calls = Config.default_dyer
    print(f"\n{logger}.sinks = {logger.instance().sinks()}")
    logger.trace(msg)
    logger.debug(msg)
    logger.info(msg)
    logger.notice(msg)
    logger.warn(msg)
    logger.error(msg)
    logger.critical(msg)
    logger.alert(msg)
    logger.fatal(msg)
    print(f"\n{logger}.level = {logger.level()}")

    objlogger = logger.objLogger(logger)
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
