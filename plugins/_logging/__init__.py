from typing import Callable


__all__ = ["Color", "Logger", "PrefixLogger", "Config", "logger"]


class Color:
    # fmt: off
    class ctl:
        reset: str                          = "\33[0m"
        bold: str                           = "\33[1m"
        dim: str                            = "\33[2m"
        italic: str                         = "\33[3m"
        underline: str                      = "\33[4m"
        blink: str                          = "\33[5m"
        blink2: str                         = "\33[6m"
        inverse: str                        = "\33[7m"
        hidden: str                         = "\33[8m"
        strikethrough: str                  = "\33[9m"
        bold_off: str                       = "\33[21m"
        dim_off: str                        = "\33[2m"
        italic_off: str                     = "\33[3m"
        underline_off: str                  = "\33[4m"
        blink_off: str                      = "\33[5m"
        blink2_off: str                     = "\33[6m"
        inverse_off: str                    = "\33[7m"
        hidden_off: str                     = "\33[8m"
        strikethrough_off: str              = "\33[9m"
        clear_screen: str                   = "\033[2J"
        clear_cursor_to_begin: str          = "\033[1J"
        clear_cursor_to_end: str            = "\033[0J"
        clear_line: str                     = "\033[2K"
        clear_cursor_to_line_start: str     = "\033[1K"
        clear_cursor_to_line_end: str       = "\033[0K"
    class cursor:
        @staticmethod
        def up(n: int) -> str:                 return f"\033[{n}A"
        @staticmethod
        def down(n: int) -> str:               return f"\033[{n}B"
        @staticmethod
        def forward(n: int) -> str:            return f"\033[{n}C"
        @staticmethod
        def backward(n: int) -> str:           return f"\033[{n}D"
        @staticmethod
        def line_start_down(n: int) -> str:    return f"\033[{n}E"
        @staticmethod
        def line_start_up(n: int) -> str:      return f"\033[{n}F"
        @staticmethod
        def column(n: int) -> str:             return f"\033[{n}G"
        @staticmethod
        def row_column(r: int, c: int) -> str: return f"\033[{r};{c}H"
        save: str        = "\033[s"
        restore: str     = "\033[u"
        hide: str        = "\033[?25l"
        show: str        = "\033[?25h"
    class fg:
        black: str       = "\33[30m"
        red: str         = "\33[31m"
        green: str       = "\33[32m"
        yellow: str      = "\33[33m"
        blue: str        = "\33[34m"
        magenta: str     = "\33[35m"
        cyan: str        = "\33[36m"
        white: str       = "\33[37m"
        reset: str       = "\33[39m"
        lblack: str      = "\33[90m"
        lred: str        = "\33[91m"
        lgreen: str      = "\33[92m"
        lyellow: str     = "\33[93m"
        lblue: str       = "\33[94m"
        lmagenta: str    = "\33[95m"
        lcyan: str       = "\33[96m"
        lwhite: str      = "\33[97m"
        @staticmethod
        def c256(n: int):
            if n > 255:
                raise ValueError(n)
            return f"\033[38;5;{n}m"
        @staticmethod
        def rgb(r: int, g: int, b: int):
            if r > 255 or g > 255 or b > 255:
                raise ValueError(r, g, b)
            return f"\033[38;2;{r};{g};{b}m"
        @classmethod
        def hex(cls, s: str):
            if len(s) == 7 and s[0] == "#":
                h = s[1:]
            elif len(s) == 6:
                h = s
            else:
                raise ValueError(s)
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return cls.rgb(r, g, b)
    class bg:
        black: str       = "\33[40m"
        red: str         = "\33[41m"
        green: str       = "\33[42m"
        yellow: str      = "\33[43m"
        blue: str        = "\33[44m"
        magenta: str     = "\33[45m"
        cyan: str        = "\33[46m"
        white: str       = "\33[47m"
        reset: str       = "\33[49m"
        lblack: str      = "\33[100m"
        lred: str        = "\33[101m"
        lgreen: str      = "\33[102m"
        lyellow: str     = "\33[103m"
        lblue: str       = "\33[104m"
        lmagenta: str    = "\33[105m"
        lcyan: str       = "\33[106m"
        lwhite: str      = "\33[107m"
        @staticmethod
        def c256(n: int):
            if n > 255:
                raise ValueError(n)
            return f"\033[48;5;{n}m"
        @staticmethod
        def rgb(r: int, g: int, b: int):
            if r > 255 or g > 255 or b > 255:
                raise ValueError(r, g, b)
            return f"\033[48;2;{r};{g};{b}m"
        @classmethod
        def hex(cls, s: str):
            if len(s) == 7 and s[0] == "#":
                h = s[1:]
            elif len(s) == 6:
                h = s
            else:
                raise ValueError(s)
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return cls.rgb(r, g, b)
    # fmt: on


class Logger:
    def __init__(self) -> None:
        self.dyer_calls: Callable[[str], str] | None = None

    def objLogger(self, obj: str | object):
        return self

    def fatal(self, msg: str): ...
    def alert(self, msg: str): ...
    def critical(self, msg: str): ...
    def error(self, msg: str): ...
    def warn(self, msg: str): ...
    def notice(self, msg: str): ...
    def info(self, msg: str): ...
    def debug(self, msg: str): ...
    def trace(self, msg: str): ...

    def __str__(self):
        return f"<{self.__class__.__name__}:{id(self):x}>"

    def __repr__(self) -> str:
        return self.__str__()


try:
    from .spdlogger import PrefixLogger, Config, logger, test
except Exception:  # noqa: BLE001
    from .pylogger import PrefixLogger, Config, logger, test

if __name__ == "__main__":
    import os

    os.system("cls")
    test()
