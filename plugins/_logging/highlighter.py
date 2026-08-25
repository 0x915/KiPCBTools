
import os
import sys

from io import StringIO
from threading import Lock
from rich.text import Text
from rich.highlighter import Highlighter, _combine_regex
from rich.console import Console
from rich.theme import Theme
from rich.style import Style

__hl_default_lock = Lock()

__hl_default_style = {
    "dtype.int": "#00ff80",
    "dtype.hex": "#ff80ff",
    "dtype.float": "#80ffff",
    "dtype.char": "#404040",
    "dtype.true": "#00ff00",
    "dtype.false": "#FF0000",
    "dtype.none": "#808080",
    "dtype.ipv4": "#b7edff",
    "dtype.ipv6": "#b7edff",
    "dtype.eui48": "#b7edff",
    "dtype.eui64": "#b7edff",
    "dtype.path": "#00b7ff",
    "dtype.uuid": "#00b7ff",
    "dtype.call": "#a6e22e",
    "dtype.symbol1": "#800000",
    "dtype.symbol2": "#808000",
    "dtype.symbol3": "#008000",
    "dtype.symbol4": "#008080",
    "dtype.symbol5": "#000080",
    "dtype.symbol6": "#800080",
    "dtype.symbol7": "#FF0000",
    "dtype.symbol8": "#FFFFFF",
}

__hl_default_theme = Theme()
__hl_default_theme.styles = {name: style if isinstance(style, Style) else Style.parse(style) for name, style in __hl_default_style.items()}


class DefaultHighlighter(Highlighter):
    base_style = "dtype."

    re_Next_s = r"(?=\s)"
    re_Prev_s = r"(?<=\s)"
    re_StrStart = r"(^)"
    re_StrEnd = r"($)"

    re_Next_Bracket = r"(?=[\(\)\[\]\<\>\{\}])"
    re_Prev_Bracket = r"(?<=[\(\)\[\]\<\>\{\}])"

    re_Next_Symbol = r"(?=[\|\~\!\,\%\^\&\*\+\-\=\\\/])"
    re_Prev_Symbol = r"(?<=[\|\~\!\,\%\^\&\*\+\-\=\\\/])"

    re_token_s = rf"({re_StrStart}|{re_Prev_s}|{re_Prev_Symbol}|{re_Prev_Bracket})"
    re_token_e = rf"({re_Next_s}|{re_StrEnd}|{re_Next_Symbol}|{re_Next_Bracket})"

    highlights = [
        r"(?P<tag_start><)(?P<tag_name>[-\w.:|]*)(?P<tag_contents>[\w\W]*)(?P<tag_end>>)",
        r'(?P<attrib_name>[\w_]{1,50})=(?P<attrib_value>"?[\w_]+"?)?',
        r"(?P<brace>[][{}()])",
        _combine_regex(
            # r"(?P<ipv4>[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}([//][0-9]{1,2})?([:][0-9]{1,6})?)",
            # r"(?P<ipv6>([A-Fa-f0-9]{1,4}::?){1,7}[A-Fa-f0-9]{1,4})"
            # r"(?P<uuid>[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})",
            # r"(?P<call>[\w.]*?)\(",
            re_token_s + r"(?P<true>[Tt]rue)" + re_token_e,
            re_token_s + r"(?P<false>[Ff]alse)" + re_token_e,
            re_token_s + r"(?P<none>[Nn]one)" + re_token_e,
            # r"(?P<path>([^<>:\"/\\|?*]+:|[.]+|[^<>:\"/\\|?*]+)?(([\\]|[\/\/])+([^<>:\"/\\|?*])+)+)",
            re_token_s + r"(?P<hex>0x[0-9a-fA-F]+)" + re_token_e,
            re_token_s + r"(?P<float>([0-9]+\.[0-9]+))" + re_token_e,
            re_token_s + r"(?P<int>[0-9]+)" + re_token_e,
            r"(?P<symbol1>[/(/)]+)",
            r"(?P<symbol2>[/[/]]+)",
            r"(?P<symbol3>[/{/}]+)",
            r"(?P<symbol4>[/</>]+)",
            r"(?P<symbol5>(:|;|^)+)",
            r"(?P<symbol6>(\"|\')+)",
            r"(?P<symbol7>(\+|\-|\*|\\|/|\||=)+)",
            # r"(?P<symbol4>[,.?:;'\"\\\/|]+)",
            r"(?P<char>[^\s])",
        ),
    ]

    def highlight(self, text: Text) -> None:
        highlight_regex = text.highlight_regex
        for re_highlight in self.highlights:
            highlight_regex(re_highlight, style_prefix=self.base_style)


__hl_default_highlighter = DefaultHighlighter()

__hl_stringio = Console(
    color_system="truecolor",
    file=StringIO(),
    theme=__hl_default_theme,
    highlighter=__hl_default_highlighter,
    soft_wrap=True,
)
__hl_console = Console(
    color_system="truecolor",
    file=sys.stdout,
    theme=__hl_default_theme,
    highlighter=__hl_default_highlighter,
    soft_wrap=True,
)


def default_highlight_str(s: str) -> str:
    with __hl_default_lock as _:
        __hl_stringio.file = StringIO()
        __hl_stringio.print(s + " ", end="")
        return str(__hl_stringio.file.getvalue())  # type: ignore


default_console = __hl_console


def __highlight_test():
    print(default_highlight_str("1.00 01.00 +1.00 -1.00 #1.00 1.00# 1.00"))
    print(default_highlight_str("(1,1) (+1,-1) (1.0,1.0) (+1.0,-1.0)"))
    print(default_highlight_str("{1,1} {+1,-1} {1.0,1.0} {+1.0,-1.0}"))
    print(default_highlight_str("[1,1] [+1,-1] [1.0,1.0] [+1.0,-1.0]"))
    print(default_highlight_str("<1,1> <+1,-1> <1.0,1.0> <+1.0,-1.0>"))
    print(default_highlight_str("1 01 +1 -1 #1 1"))
    print(default_highlight_str("0x0123456789ABCDEF 0xABC0123456789DEF ABC0123456789DEF 0x0123456789ABCDEF"))
    print(default_highlight_str("~!@#$%^&*_+`-="))
    print(default_highlight_str("1234567890 qwer QWER ,.?()[]{}<>:;'\"\\/|"))
    print(default_highlight_str("True true False false None none"))

if __name__ == "__main__":
    os.system("cls")
    __highlight_test()