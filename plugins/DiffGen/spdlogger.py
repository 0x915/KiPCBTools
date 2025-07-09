from enum import IntEnum

import spdlog  # type: ignore

print()


class LoggerUtil:
    class COLOR:
        class CTL:
            END = "\33[0m"
            BOLD = "\33[1m"
            ITALIC = "\33[3m"
            URL = "\33[4m"
            BLINK = "\33[5m"
            BLINK2 = "\33[6m"
            SELECTED = "\33[7m"

        class FG:
            BLACK = "\33[30m"
            RED = "\33[31m"
            GREEN = "\33[32m"
            YELLOW = "\33[33m"
            BLUE = "\33[34m"
            MAGENTA = "\33[35m"
            BEIGE = "\33[36m"
            WHITE = "\33[37m"
            LBLACK = "\33[90m"
            LRED = "\33[91m"
            LGREEN = "\33[92m"
            LYELLOW = "\33[93m"
            LBLUE = "\33[94m"
            LMAGENTA = "\33[95m"
            LCYAN = "\33[96m"
            LWHITE = "\33[97m"
            LDEFAULT = "\33[99m"

        class BG:
            BLACK = "\33[40m"
            RED = "\33[41m"
            GREEN = "\33[42m"
            YELLOW = "\33[43m"
            BLUE = "\33[44m"
            MAGENTA = "\33[45m"
            BEIGE = "\33[46m"
            WHITE = "\33[47m"

            LBLACK = "\33[100m"
            LRED = "\33[101m"
            LGREEN = "\33[102m"
            LYELLOW = "\33[103m"
            LBLUE = "\33[104m"
            LMAGENTA = "\33[105m"
            LCYAN = "\33[106m"
            LWHITE = "\33[107m"

            LDEFAULT = "\33[109m"

    class LEVEL(IntEnum):
        OFF = 100
        FATAL = 60
        ERROR = 50
        WARN = 40
        INFO = 30
        DEBUG = 20
        TRACK = 10

    @staticmethod
    def StdoutSink(
        level: LEVEL = LEVEL.TRACK,
    ):  # , pattern: str="\033[90m[%C-%m-%d %H:%M:%S %f %L]\033[0m %^%v%$"):
        sink = spdlog.stdout_sink_mt()
        sink.set_level(level)
        # sink.set_pattern(pattern)
        # sink.set_color(LoggerUtil.LEVEL.OFF, LoggerUtil.COLOR.FG.BLACK + LoggerUtil.COLOR.BG.RED)
        # sink.set_color(LoggerUtil.LEVEL.ERROR, LoggerUtil.COLOR.FG.RED)
        # sink.set_color(LoggerUtil.LEVEL.WARN, LoggerUtil.COLOR.FG.YELLOW)
        # sink.set_color(LoggerUtil.LEVEL.INFO, LoggerUtil.COLOR.FG.LGREEN)
        # sink.set_color(LoggerUtil.LEVEL.DEBUG, LoggerUtil.COLOR.FG.LBLACK)
        # sink.set_color(LoggerUtil.LEVEL.TRACK, LoggerUtil.COLOR.FG.LBLUE)
        return sink

    @staticmethod
    def ColorStdoutSink(
        level: LEVEL = LEVEL.TRACK,
    ):  # , pattern: str="\033[90m[%C-%m-%d %H:%M:%S %f %L]\033[0m %^%v%$"):
        sink = spdlog.stdout_color_sink_mt()
        sink.set_level(level)
        # sink.set_pattern(pattern)
        # sink.set_color(LoggerUtil.LEVEL.OFF, LoggerUtil.COLOR.FG.BLACK + LoggerUtil.COLOR.BG.RED)
        # sink.set_color(LoggerUtil.LEVEL.ERROR, LoggerUtil.COLOR.FG.RED)
        # sink.set_color(LoggerUtil.LEVEL.WARN, LoggerUtil.COLOR.FG.YELLOW)
        # sink.set_color(LoggerUtil.LEVEL.INFO, LoggerUtil.COLOR.FG.LGREEN)
        # sink.set_color(LoggerUtil.LEVEL.DEBUG, LoggerUtil.COLOR.FG.LBLACK)
        # sink.set_color(LoggerUtil.LEVEL.TRACK, LoggerUtil.COLOR.FG.LBLUE)
        return sink

    @staticmethod
    def DailyFileSink(filename: str, level: LEVEL = LEVEL.TRACK):  # , pattern: str="[%C-%m-%d %H:%M:%S %f][%L]%v%$"):
        sink = spdlog.daily_file_sink_mt(filename, 0, 0)
        sink.set_level(level)
        # sink.set_pattern(pattern)
        return sink

    COLOR_END = COLOR.CTL.END
    COLOR_FATAL = COLOR.FG.BLACK + COLOR.BG.MAGENTA
    COLOR_ERROR = COLOR.FG.RED
    COLOR_WARN = COLOR.FG.YELLOW
    COLOR_INFO = COLOR.FG.LGREEN
    COLOR_DEBUG = COLOR.FG.LBLACK
    COLOR_TRACK = COLOR.FG.LBLUE

    class NoneLogger:
        # fmt:off
        @staticmethod
        def fatal(msg: str): pass
        @staticmethod
        def error(msg: str): pass
        @staticmethod
        def warn(msg: str): pass
        @staticmethod
        def info(msg: str): pass
        @staticmethod
        def debug(msg: str): pass
        @staticmethod
        def track(msg: str): pass
        # fmt:on

    class Logger:
        def __init__(self, name_or_logger: str | spdlog.SinkLogger | None, sinks: list) -> None:
            self._prefix: str = ""
            self._logger: spdlog.SinkLogger | None = None
            self._log = self._null

            if isinstance(name_or_logger, spdlog.SinkLogger):
                self.SetLogger(name_or_logger)
                return

            if isinstance(name_or_logger, str):
                self._init(name_or_logger, sinks)
                return

            print(f"警告:日志记录器(0x{id(self):X})没有活动实例.")
            return

        def _init(self, name: str, sinks: list):
            self._logger = spdlog.SinkLogger(
                name,
                sinks if len(sinks) != 0 else [LoggerUtil.StdoutSink()],
            )
            assert self._logger is not None
            self._logger.set_pattern("\033[90m[%C%m%d %H:%M:%S %e][%n]\033[0m %v", spdlog.local)
            self._log = self._logger.log
            return

        def Disable(self):
            self._log = self._null

        def Enable(self):
            if self._logger:
                self._log = self._logger.log
                return
            self._init("NULL", [])

        def GetLogger(self):
            return self._logger

        def SetLogger(self, obj: spdlog.SinkLogger):
            self._logger = obj
            self._log = self._logger.log

        def ObjLogger(self, obj: str | object):
            if not isinstance(obj, str):
                name = f"0x{id(obj):x}"
            else:
                name = obj

            if self._logger:
                get = LoggerUtil.Logger(self._logger.name(), self._logger.sinks())
                get._prefix = "<" + name + "> "
                return get

            return LoggerUtil.Logger("", None)

        def _null(self, level: "LoggerUtil.LEVEL", msg: str):
            pass

        def fatal(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.FATAL,
                f"[F] {LoggerUtil.COLOR_FATAL} {self._prefix}{msg} {LoggerUtil.COLOR_END}",
            )

        def error(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.ERROR,
                f"[E] {LoggerUtil.COLOR_ERROR}{self._prefix}{msg}{LoggerUtil.COLOR_END}",
            )

        def warn(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.WARN,
                f"[W] {LoggerUtil.COLOR_WARN}{self._prefix}{msg}{LoggerUtil.COLOR_END}",
            )

        def info(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.INFO,
                f"[I] {LoggerUtil.COLOR_INFO}{self._prefix}{msg}{LoggerUtil.COLOR_END}",
            )

        def debug(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.DEBUG,
                f"[D] {LoggerUtil.COLOR_DEBUG}{self._prefix}{msg}{LoggerUtil.COLOR_END}",
            )

        def track(self, msg: str):
            self._log(
                LoggerUtil.LEVEL.TRACK,
                f"[T] {LoggerUtil.COLOR_TRACK}{self._prefix}{msg}{LoggerUtil.COLOR_END}",
            )


def main():
    sinks = [
        LoggerUtil.StdoutSink(),
        LoggerUtil.DailyFileSink("test.log"),
    ]

    logger = LoggerUtil.Logger("main", sinks)

    logger.track("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")
    logger.debug("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")
    logger.info("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")
    logger.warn("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")
    logger.error("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")
    logger.fatal("~!@#$%^&*()_+|`1234567890-=|qwer|QWER|[];',./|{{}}:\"<>?")

if __name__ == "__main__":
    main()
