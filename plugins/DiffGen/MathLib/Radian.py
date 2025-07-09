import math
import copy
from math import radians

from MathLib.Define import Num, StrNum


class Rad:
    DEF_0DEG = 0
    DEF_90DEG = 0.5 * math.pi
    DEF_180DEG = math.pi
    DEF_270DEG = 1.5 * math.pi
    DEF_360DEG = 2 * math.pi

    # region ## 构造函数

    def __init__(self, rad: Num) -> None:
        self._v: Num = 0
        self.Set(rad)

    @staticmethod
    def fromDeg(v: Num) -> "Rad":
        return Rad(rad=math.radians(v))

    def Clone(self) -> "Rad":
        return copy.copy(self)

    # endregion
    # region ## 私有成员代理

    def Get(self):
        return self._v

    def Set(self, rad: Num):
        self._v = Rad.to360(rad)

    def GetDeg(self) -> Num:
        return math.degrees(self._v)

    def SetDeg(self, v: Num) -> None:
        self.Set(math.radians(v))

    @staticmethod
    def to360(v: Num) -> Num:
        if v == Rad.DEF_360DEG:
            return 0
        if v < 0:
            return v % -(2 * math.pi)
        else:
            return v % (2 * math.pi)

    def toPositive(self) -> None:
        if self._v >= 0:
            return
        self.Set(2 * math.pi + self._v)

    def toNegative(self) -> None:
        if self._v <= 0:
            return
        self.Set(self._v - 2 * math.pi)

    # endregion

    # region ## 角度特性

    def cos(self) -> float:
        return math.cos(self._v)

    def sin(self) -> float:
        return math.sin(self._v)

    def tan(self) -> float:
        return math.tan(self._v)

    def acos(self) -> float:
        return math.acos(self._v)

    def asin(self) -> float:
        return math.asin(self._v)

    def atan(self) -> float:
        return math.atan(self._v)

    # endregion

    # region ## 内置方法覆盖

    def __abs__(self, value: "Rad") -> "Rad":
        return Rad(abs(self._v))

    def __neg__(self) -> "Rad":
        return Rad((2 * math.pi) - self._v)

    PRINT_FLOAT_DIGIT_NUM = 6

    def __str__(self) -> str:
        DIG = self.PRINT_FLOAT_DIGIT_NUM
        return f"<{self.__class__.__name__} ({StrNum(self.Get(), DIG)}pi) DEG({StrNum(self.GetDeg(), DIG)}) 0x{id(self):X}>"

    def __lt__(self, value: "Rad") -> bool:
        return self._v < value._v

    def __le__(self, value: "Rad") -> bool:
        return self._v <= value._v

    def __eq__(self, value: "Rad") -> bool:  # type: ignore
        return self._v == value._v

    def __ne__(self, value: "Rad") -> bool:  # type: ignore
        return self._v != value._v

    def __gt__(self, value: "Rad") -> bool:
        return self._v > value._v

    def __ge__(self, value: "Rad") -> bool:
        return self._v >= value._v

    def __add__(self, value: "Rad") -> "Rad":
        return Rad(self._v + value._v)

    def __sub__(self, value: "Rad") -> "Rad":
        return Rad(self._v - value._v)

    def __truediv__(self, value: "Rad") -> float:
        return self._v / value._v

    def __repr__(self) -> str:
        return self.__str__()

    # endregion


G_RAD_0_DEG = Rad.fromDeg(0)
G_RAD_90_DEG = Rad.fromDeg(90)
G_RAD_180_DEG = Rad.fromDeg(180)
G_RAD_270_DEG = Rad.fromDeg(270)

