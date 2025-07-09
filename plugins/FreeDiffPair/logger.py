from typing import TextIO

import logging
import sys


class color:
    reset = "\33[0m"
    bold = "\33[2m"
    italic = "\33[3m"
    underline = "\33[4m"

    f_black = "\33[30m"
    f_red = "\33[31m"
    f_green = "\33[32m"
    f_yellow = "\33[33m"
    f_blue = "\33[34m"
    f_magenta = "\33[35m"
    f_cyan = "\33[36m"
    f_white = "\33[37m"
    f_default = "\33[39m"

    b_black = "\33[40m"
    b_red = "\33[41m"
    b_green = "\33[42m"
    b_yellow = "\33[43m"
    b_blue = "\33[44m"
    b_magenta = "\33[45m"
    b_cyan = "\33[46m"
    b_white = "\33[47m"
    b_default = "\33[49m"

    f_blackL = "\33[90m"
    f_redL = "\33[91m"
    f_greenL = "\33[92m"
    f_yellowL = "\33[93m"
    f_blueL = "\33[94m"
    f_magentaL = "\33[95m"
    f_cyanL = "\33[96m"
    f_whiteL = "\33[97m"
    f_defaultL = "\33[99m"

    b_blackL = "\33[100m"
    b_redL = "\33[101m"
    b_greenL = "\33[102m"
    b_yellowL = "\33[103m"
    b_blueL = "\33[104m"
    b_magentaL = "\33[105m"
    b_cyanL = "\33[106m"
    b_whiteL = "\33[107m"
    b_defaultL = "\33[109m"

    end = "\33[0m"

    class Style:
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
        VIOLET = "\33[35m"
        BEIGE = "\33[36m"
        WHITE = "\33[37m"

    class BG:
        BLACK = "\33[40m"
        RED = "\33[41m"
        GREEN = "\33[42m"
        YELLOW = "\33[43m"
        BLUE = "\33[44m"
        VIOLET = "\33[45m"
        BEIGE = "\33[46m"
        WHITE = "\33[47m"


TOP = 100
FATAL = 75
CRITICAL = 65
ERROR = 55
WARN = 45
INFO = 35
DEBUG = 25
TRACK = 15

logging.addLevelName(TOP, "#")
logging.addLevelName(FATAL, "F")
logging.addLevelName(CRITICAL, "C")
logging.addLevelName(ERROR, "E")
logging.addLevelName(WARN, "W")
logging.addLevelName(INFO, "I")
logging.addLevelName(DEBUG, "D")
logging.addLevelName(TRACK, "T")

NOTSET = 0


class DEFAULT_CONFIG:
    LOG_LEVEL = NOTSET
    DATE_FORMAT: str = "%y%m%d %H:%M:%S"
    FMT_STREAM: str = "[%(asctime)s %(msecs)3d][%(levelname)1s] <%(name)s> %(message)s"
    FMT_FILE: str = "[%(asctime)s.%(msecs)3d][%(levelname)1s] <%(name)s> %(message)s"
    STREAM: TextIO | None = sys.stdout

class EnhancedLogger(logging.Logger):
    class _RE_COLOR:
        def __init__(self) -> None:
            self.INT = color.f_greenL
            self.FLOAT = color.f_cyanL
            self.AZ = color.f_blackL
            self.BRACKET = color.f_red
            self.HEX = color.f_blueL
            self.SYM = color.f_redL
            self.MATH = color.f_magentaL
            self.END = color.end

    def __init__(self, name: str, level: int | str = 0) -> None:
        super().__init__(name, level)

        self.COLORS = EnhancedLogger._RE_COLOR()

    def _custom_log(self, level: int, msg: str):
        TYPE_NUM = 2
        TYPE_HEX = 4

        def add_color(ch_type: int, tmp: str):
            if ch_type == 0:
                return tmp
            if ch_type == 1:
                return self.COLORS.AZ + tmp + self.COLORS.END
            if ch_type == TYPE_NUM:
                if "." in tmp:
                    return self.COLORS.FLOAT + tmp + self.COLORS.END
                else:
                    return self.COLORS.INT + tmp + self.COLORS.END
            if ch_type == 3:
                return self.COLORS.BRACKET + tmp + self.COLORS.END
            if ch_type == TYPE_HEX:
                return self.COLORS.HEX + tmp + self.COLORS.END
            if ch_type == 5:
                return self.COLORS.MATH + tmp + self.COLORS.END
            return tmp

        def color_char(strin: str):
            ch_type: int = 0
            sym_mode: bool = False
            out_msg: str = ""
            tmp: str = ""

            for ch in strin:
                i = ord(ch)
                # 1:大小字母
                if (i >= 65 and i <= 90) or (i >= 97 and i <= 122):
                    if ch_type != 1:
                        out_msg += add_color(ch_type, tmp)
                        tmp = ""
                        # 字母默认开启连写模式 直到遇到 非字母数字
                        sym_mode = True
                        ch_type = 1
                    tmp += ch
                    continue
                # 2:整型浮点
                elif (i >= 48 and i <= 57) or (ch_type == 2 and ch == "."):
                    # 符号连续匹配
                    if ch_type == 1 and sym_mode:
                        tmp += ch
                        continue
                    if ch_type != 2:
                        out_msg += add_color(ch_type, tmp)
                        tmp = ""
                        ch_type = 2
                    tmp += ch
                    continue
                elif ch in "{}(),<>[]":
                    sym_mode = False
                    if ch_type != 3:
                        out_msg += add_color(ch_type, tmp)
                        tmp = ""
                        ch_type = 3
                    tmp += ch
                    continue
                elif ch in "+-*/=^%&#?:!~":
                    sym_mode = False
                    if ch_type != 5:
                        out_msg += add_color(ch_type, tmp)
                        tmp = ""
                        ch_type = 5
                    tmp += ch
                    continue
                # 字母后面允许连写数字
                elif ch in "_":
                    sym_mode = True
                    tmp += ch
                # 0:其他字符
                else:
                    sym_mode = False
                    if ch_type != 0:
                        out_msg += add_color(ch_type, tmp)
                        tmp = ""
                        ch_type = 0
                    tmp += ch
                    continue

            out_msg += add_color(ch_type, tmp)
            return out_msg

        # 只有 (0x) 开头 [0-9] [AZ] [az]
        def isHex(strin: str):
            if not (strin[0] == "0" and strin[1] == "x"):
                return False
            for ch in strin[2:]:
                i = ord(ch)
                if (i >= 65 and i <= 70) or (i >= 97 and i <= 102) or (i >= 48 and i <= 57):
                    continue
                return False
            return True

        # 只有 [0-9] 和 [.]
        def isNum(strin: str):
            for ch in strin:
                i = ord(ch)
                if i == 46 or (i >= 48 and i <= 57):
                    continue
                return False
            return True

        color_msg: str = ""
        for s in msg.split(" "):
            if s == "":
                color_msg += " "
                continue
            if isHex(s):
                color_msg += add_color(TYPE_HEX, s) + " "
                continue
            if isNum(s):
                color_msg += add_color(TYPE_NUM, s) + " "
                continue
            color_msg += color_char(s) + " "

        self._log(level, color_msg, ())

    def track(self, msg: str):
        if not self.isEnabledFor(TRACK):
            return
        self._custom_log(TRACK, msg)

    def debug(self, msg: str):  # type: ignore
        if not self.isEnabledFor(DEBUG):
            return
        self._custom_log(DEBUG, msg)

    def info(self, msg: str):  # type: ignore
        if not self.isEnabledFor(INFO):
            return
        self._custom_log(INFO, msg)

    def warn(self, msg: str):  # type: ignore
        if not self.isEnabledFor(WARN):
            return
        self._custom_log(WARN, msg)

    def error(self, msg: str):  # type: ignore
        if not self.isEnabledFor(ERROR):
            return
        self._custom_log(ERROR, msg)

    def critical(self, msg: str):  # type: ignore
        if not self.isEnabledFor(CRITICAL):
            return
        self._custom_log(CRITICAL, msg)

    def fatal(self, msg: str):
        if not self.isEnabledFor(FATAL):
            return
        self._custom_log(FATAL, msg)

    def top(self, msg: str):
        if not self.isEnabledFor(TOP):
            return
        self._custom_log(TOP, msg)

    def addFileHandler(
        self,
        filename: str,
        mode: str = "a",
        encoding: str = "UTF-8",
        fmt: str = DEFAULT_CONFIG.FMT_FILE,
        datefmt: str = DEFAULT_CONFIG.DATE_FORMAT,
    ):
        handler = logging.FileHandler(filename=filename, mode=mode, encoding=encoding)
        handler.setLevel(self.getEffectiveLevel())
        handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt=datefmt)
        )

        for h in self.handlers:
            if not isinstance(h, logging.FileHandler):
                continue
            if h.baseFilename == handler.baseFilename:
                return False
            continue

        self.addHandler(handler)
        return True

    def addStreamHandler(
        self,
        stream=DEFAULT_CONFIG.STREAM,
        fmt: str = DEFAULT_CONFIG.FMT_STREAM,
        datefmt: str = DEFAULT_CONFIG.DATE_FORMAT,
    ):
        stdout_handler = logging.StreamHandler(stream)
        stdout_handler.setLevel(self.getEffectiveLevel())

        stdout_handler.setFormatter(

            logging.Formatter(fmt=fmt, datefmt=datefmt)
        )

        for h in self.handlers:
            if not isinstance(h, logging.StreamHandler):
                continue
            if h.stream == stdout_handler.stream:
                return False

        self.addHandler(stdout_handler)
        return True


_root = EnhancedLogger("root", DEFAULT_CONFIG.LOG_LEVEL)
# _root.addStreamHandler(sys.stdout)
EnhancedLogger.root = _root  # type: ignore
EnhancedLogger.manager = logging.Manager(_root)  # type: ignore
EnhancedLogger.manager.loggerClass = EnhancedLogger
logger = _root


def getLogger(obj: object | str | None = None) -> EnhancedLogger:
    if obj is None:
        return _root
    if isinstance(obj, str):
        if obj == _root.name:
            return _root
        return EnhancedLogger.manager.getLogger(obj)  # type: ignore
    obj = f"{str(type(obj))[8:-2]}:{id(obj):X}"
    return EnhancedLogger.manager.getLogger(obj)  # type: ignore
