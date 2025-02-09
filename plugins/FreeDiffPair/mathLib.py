import math
from typing import List, Tuple


class Point2D:
    def __init__(self, x: int | float, y: int | float) -> None:
        self.x = x
        self.y = y

    ##############################

    def toVector2D(self):
        return Vec2D(self.x, self.y)

    def toTuple(self) -> Tuple[int | float, int | float]:
        return (self.x, self.y)

    ##############################

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, value: "Point2D") -> "Point2D":
        return Point2D(self.x + value.x, self.y + value.y)

    def __sub__(self, value: "Point2D") -> "Point2D":
        return Point2D(self.x - value.x, self.y - value.y)

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y  # type: ignore

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value=value)


G_ANGLE_RAD_TOLERANCE = 0.0000005
G_ANGLE_DEG_FLOAT_DIGITS = 12


class Rad:
    PRINT_FLOAT_DIGIT_NUM = 6

    DEF_0DEG = 0
    DEF_90DEG = 0.5 * math.pi
    DEF_180DEG = math.pi
    DEF_270DEG = 1.5 * math.pi
    DEF_360DEG = 2 * math.pi

    def __init__(self, rad: int | float) -> None:
        self._v = self.toOneCycle(rad)

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self.toOneCycle(v)

    @staticmethod
    def fromDeg(v: int | float):
        return Rad(rad=math.radians(v))

    def cvtPositive(self):
        if self._v >= 0:
            return
        self._v = 2 * math.pi + self._v

    def cvtNegative(self):
        if self._v <= 0:
            return
        self._v = self._v - 2 * math.pi

    def SetFromDeg(self, v: int | float):
        self.value = math.radians(v)

    def toDegFloat(self, digits=G_ANGLE_DEG_FLOAT_DIGITS):
        return round(math.degrees(self._v), digits)

    def cos(self):
        return math.cos(self._v)

    def sin(self):
        return math.sin(self._v)

    def tan(self):
        return math.tan(self._v)

    def acos(self):
        return math.acos(self._v)

    def asin(self):
        return math.asin(self._v)

    def atan(self):
        return math.tan(self._v)

    @staticmethod
    def toOneCycle(v: int | float):
        if v < 0:
            return v % -(2 * math.pi)
        else:
            return v % (2 * math.pi)

    def __abs__(self, value: "Rad"):
        return Rad(abs(self._v))

    def __neg__(self):
        return (2 * math.pi) - self._v

    def __str__(self) -> str:
        FD = self.PRINT_FLOAT_DIGIT_NUM
        return f"Rad({self._v / math.pi:+.{FD}f}pi, {self.toDegFloat():+.{FD}f}deg)"

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

    def __add__(self, value: "Rad"):
        return Rad(self._v + value._v)

    def __sub__(self, value: "Rad"):
        return Rad(self._v - value._v)

    def __truediv__(self, value: "Rad"):
        return self._v / value._v

    def __mod__(self, value: "Rad"):
        return Rad(self._v % value._v)

    def __repr__(self) -> str:
        return self.__str__()


G_RAD_0_DEG = Rad.fromDeg(0)
G_RAD_90_DEG = Rad.fromDeg(90)
G_RAD_180_DEG = Rad.fromDeg(180)
G_RAD_270_DEG = Rad.fromDeg(270)


class Vec3D:
    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.bias: List[Vec3D] = []

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @staticmethod
    def CrossProduct(u: "Vec3D", v: "Vec3D"):
        u1, u2, u3 = u.x, u.y, u.z
        v1, v2, v3 = v.x, v.y, v.z
        x = u2 * v3 - u3 * v2
        y = u1 * v3 + u3 * v1
        z = u1 * v2 - u2 * v1
        return Vec3D(x, -y, z)


class Vec2D:
    PRINT_FLOAT_DIGIT_NUM = 4

    def __init__(self, x: int | float, y: int | float, bias: List["Vec2D"] = []) -> None:
        self.x = x
        self.y = y
        self.biasList = bias

    ##############################

    @property
    def bias(self):
        u = Vec2D(0, 0)
        for v in self.biasList:
            u += v
        return u

    def SetBias(self, vList: List["Vec2D"]):
        self.biasList = vList

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def SetNorm(self, s: int | float) -> "Vec2D":
        ratio = s / self.norm
        return Vec2D(self.x * ratio, self.y * ratio)

    def IsZero(self):
        return self.x == 0 and self.y == 0

    @property
    def angle(self):
        if self.IsZero():
            raise ValueError("零向量角度无法确定")

        ref = Vec2D(1, 0)
        r = math.acos(
            # (x1*y1 + x2*y2)
            (self.x * ref.x + self.y * ref.y)
            # ---------------
            # |self| * |(1,0)|
            / (self.norm * ref.norm)
        )
        if self.y < 0:
            r = -r
        rad = Rad(r)
        rad.cvtPositive()
        return rad

    def SetAngle(self, v: Rad) -> "Vec2D":
        return self.Rotate(v - self.angle)

    @property
    def withBias(self):
        return self.bias + self

    ##############################

    def toVec3D(self):
        return Vec3D(self.x, self.y, 0)

    def toTuple(self) -> Tuple[int | float, int | float]:
        return (self.x, self.y)

    def Clone(self):
        return Vec2D(self.x, self.y)

    def Rotate(self, v: Rad):
        cos = v.cos()
        sin = v.sin()
        x = self.x * cos - self.y * sin
        y = self.x * sin + self.y * cos
        return Vec2D(x, y)

    ##############################

    @staticmethod
    def AngleEqual(a: "Vec2D", b: "Vec2D", rad_tolerance=G_ANGLE_RAD_TOLERANCE):
        if abs(Vec2D.GetIncludedAngle(a, b).value) <= rad_tolerance:
            return True
        return False

    @staticmethod
    def GetIncludedAngle(a: "Vec2D", b: "Vec2D"):
        return b.angle - a.angle

    @staticmethod
    def isParallel(a: "Vec2D", b: "Vec2D", tolerance=0.0001, debuger=None) -> int:
        """判断 向量(a) 与 向量(b)平行：正值(+)同向平行，负值(-)反向平行，零值不平行"""

        if a.IsZero() or b.IsZero():
            return 1

        aU = a.SetNorm(1)
        bU = b.SetNorm(1)
        
        c = aU - bU
        if c.norm <= tolerance:
            return 1

        c = aU + bU
        if c.norm <= tolerance:
            return -1
            
        return 0


    @staticmethod
    def GetParallelDistance(u: "Vec2D", v: "Vec2D"):
        w = v.bias - u.bias
        aW = w.angle
        aU = u.angle
        a = aU - aW
        return w.norm * a.sin()

    @staticmethod
    def GetLinearJunction(u: "Vec2D", v: "Vec2D"):
        x1, y1 = u.bias.toTuple()
        x2, y2 = u.withBias.toTuple()
        x3, y3 = v.bias.toTuple()
        x4, y4 = v.withBias.toTuple()
        x_up = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        x_dn = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if x_dn == 0:
            return None
        y_up = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        y_dn = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if y_dn == 0:
            return None
        return Vec2D(x_up / x_dn, y_up / y_dn)

    @staticmethod
    def DirectedArea(a: "Vec2D", b: "Vec2D"):
        return a.x * b.y - b.x * a.y

    @staticmethod
    def CrossProduct(u: "Vec2D", v: "Vec2D"):
        return Vec3D.CrossProduct(u.toVec3D(), v.toVec3D())

    @staticmethod
    def DotProduct(a, b: "Vec2D"):
        return a.norm * b.norm * Vec2D.GetIncludedAngle(a, b).cos()

    ##############################

    def __str__(self) -> str:
        bias = self.bias
        zero = self.IsZero()
        FD = self.PRINT_FLOAT_DIGIT_NUM
        return (
            f"{self.__class__.__name__}({self.x:+}, {self.y:+})"
            #
            + "{"
            + f"{self.angle if not zero else 'Angle(Unknown)'}, "
            + f"Size({self.norm:+.{FD}f}), "
            + f"Bias({bias.x:+}, {bias.y:+})"
            + "}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, value: int | float):
        return Vec2D(self.x * value, self.y * value)

    def __rmul__(self, value: int | float):
        return self.__mul__(value)

    def __add__(self, value: "Vec2D") -> "Vec2D":
        return Vec2D(self.x + value.x, self.y + value.y)

    def __sub__(self, value: "Vec2D") -> "Vec2D":
        return Vec2D(self.x - value.x, self.y - value.y)

    def __neg__(self) -> "Vec2D":
        return Vec2D(-self.x, -self.y)

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y  # type: ignore

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value=value)


G_LINEAR_K_TOLERANCE = 0.000001
G_LINEAR_B_TOLERANCE = 0.000001


class Linear2D:
    class LINE_TYPE_XY:
        @classmethod
        def GetX(cls, obj: "Linear2D", y: float | int):
            return (y - obj.b) / obj.k

        @classmethod
        def GetY(cls, obj: "Linear2D", x: float | int):
            return obj.k * x + obj.b

        @classmethod
        def GetJunction(cls, obj1: "Linear2D", obj2: "Linear2D"):
            if obj1.isParallel(obj2):
                return None
            x = (obj1.b - obj2.b) / (obj2.k - obj1.k)
            y = obj1.k * x + obj1.b
            return Point2D(x, y)

        @classmethod
        def GetDeltaB(cls, obj1: "Linear2D", obj2: "Linear2D"):
            b1 = obj1.b
            b2 = obj2.b
            return b2 - b1

        @classmethod
        def GetDeltaK(cls, obj1: "Linear2D", obj2: "Linear2D"):
            k1 = obj1.k
            k2 = obj2.k
            return k2 - k1

        @classmethod
        def isParallel(cls, obj1: "Linear2D", obj2: "Linear2D", k_tolerance=G_LINEAR_K_TOLERANCE):
            if abs(cls.GetDeltaK(obj1, obj2)) <= k_tolerance:
                return True
            return False

        @classmethod
        def GetParallelDistance(cls, obj1: "Linear2D", obj2: "Linear2D"):
            if not obj1.isParallel(obj2):
                return None
            if obj1.b == obj2.b:
                return 0
            b1 = obj1.b
            b2 = obj2.b
            d: int | float = abs((b1 - b2)) / ((1 + obj1.k**2) ** 0.5)
            return d

    class LINE_TYPE_X_ONLY:
        @classmethod
        def GetX(cls, obj: "Linear2D", y: float | int):
            return obj.value

        @classmethod
        def GetY(cls, obj: "Linear2D", x: float | int):
            return None

        @classmethod
        def GetJunction(cls, obj1: "Linear2D", obj2: "Linear2D"):
            # 与 垂直线 不存在交点
            if obj2.ltype == Linear2D.LINE_TYPE_X_ONLY:
                return None
            # 水平线 返回(self,)
            if obj2.ltype == Linear2D.LINE_TYPE_Y_ONLY:
                return (obj1.value, obj2.value)
            # 斜线
            x = obj1.value
            y = obj2.GetY(x)
            if y is None:
                return None
            return Point2D(x, y)

        @classmethod
        def GetDeltaB(cls, obj1: "Linear2D", obj2: "Linear2D"):
            raise ValueError

        @classmethod
        def GetDeltaK(cls, obj1: "Linear2D", obj2: "Linear2D"):
            raise ValueError

        @classmethod
        def isParallel(cls, obj1: "Linear2D", obj2: "Linear2D", k_tolerance=G_LINEAR_K_TOLERANCE):
            if obj2.ltype == Linear2D.LINE_TYPE_X_ONLY:
                return True
            return False

        @classmethod
        def GetParallelDistance(cls, obj1: "Linear2D", obj2: "Linear2D"):
            if not obj1.isParallel(obj2):
                return None
            return obj2.value - obj1.value

    class LINE_TYPE_Y_ONLY:
        @classmethod
        def GetX(cls, obj: "Linear2D", y: float | int):
            return None

        @classmethod
        def GetY(cls, obj: "Linear2D", x: float | int):
            return obj.value

        @classmethod
        def GetJunction(cls, obj1: "Linear2D", obj2: "Linear2D"):
            # 与 水平线 不存在交点
            if obj2.ltype == Linear2D.LINE_TYPE_Y_ONLY:
                return None
            # 垂直线 返回(self,)
            if obj2.ltype == Linear2D.LINE_TYPE_X_ONLY:
                return (obj2.value, obj1.value)
            # 斜线
            y = obj1.value
            x = obj2.GetX(y)
            if x is None:
                return None
            return Point2D(x, y)

        @classmethod
        def GetDeltaB(cls, obj1: "Linear2D", obj2: "Linear2D"):
            return cls.GetParallelDistance(obj1, obj2)

        @classmethod
        def GetDeltaK(cls, obj1: "Linear2D", obj2: "Linear2D"):
            return None

        @classmethod
        def isParallel(cls, obj1: "Linear2D", obj2: "Linear2D", k_tolerance=G_LINEAR_K_TOLERANCE):
            if obj2.ltype == Linear2D.LINE_TYPE_Y_ONLY:
                return True
            return False

        @classmethod
        def GetParallelDistance(cls, obj1: "Linear2D", obj2: "Linear2D"):
            if not obj1.isParallel(obj2):
                return None
            return obj2.value - obj1.value

    def __init__(self, k: int | float, b: int | float) -> None:
        self.k = k
        self.b = b
        self.value: int | float = 0
        self.ltype: Linear2D.LINE_TYPE_XY | Linear2D.LINE_TYPE_X_ONLY | Linear2D.LINE_TYPE_Y_ONLY = Linear2D.LINE_TYPE_XY()

    def SetLinearLine(self, left: Point2D, right: Point2D) -> int:
        # 同一点
        if left == right:
            raise Exception(f"{left}=={right} ???")

        self.dx = right.x - left.x
        self.dy = right.y - left.y

        # 垂直线
        if self.dx == 0:
            self.value = right.x
            self.ltype = Linear2D.LINE_TYPE_X_ONLY()
            return 1

        # 水平线
        if self.dy == 0:
            self.value = right.y
            self.ltype = Linear2D.LINE_TYPE_Y_ONLY()
            return 2

        # 斜线
        self.k = self.dy / self.dx
        self.b = left.y - (self.k * left.x)
        self.ltype = Linear2D.LINE_TYPE_XY()
        return 3

    def GetX(self, y: float | int):
        return self.ltype.GetX(self, y)

    def GetY(self, x: float | int):
        return self.ltype.GetY(self, x)

    def GetJunction(self, obj2: "Linear2D"):
        return self.ltype.GetJunction(self, obj2)

    def GetDeltaB(self, obj1: "Linear2D", obj2: "Linear2D"):
        return self.ltype.GetDeltaB(obj1, obj2)

    def GetDeltaK(self, obj1: "Linear2D", obj2: "Linear2D"):
        return self.ltype.GetDeltaK(obj1, obj2)

    def isParallel(self, obj2: "Linear2D", k_tolerance=G_LINEAR_K_TOLERANCE):
        return self.ltype.isParallel(self, obj2, k_tolerance)

    def GetParallelDistance(self, obj2: "Linear2D"):
        return self.ltype.GetParallelDistance(self, obj2)

    def __str__(self) -> str:
        return f"Linear2D(k={self.k:+.9f}, b={self.b:+.9f})"


def _Vector2D_Test():
    def TestCumulativeRotationError():
        print("------------ TestCumulativeRotationError ------------")
        vec = Vec2D(0, 1)
        rad1deg = Rad.fromDeg(-1)
        t = vec.Clone()
        for _ in range(360):
            t = t.Rotate(rad1deg)
            y2 = 1 - t.x * t.x
            d = abs(abs(t.y**2) - abs(y2))
            print(f"add {1:3d}deg = {t} Δ={d:+.18f}")

    def TestRotationError():
        print("------------ TestRotationError ------------")
        vec = Vec2D(0, 1)

        for i in range(360):
            radNdeg = Rad.fromDeg(i)
            newVec = vec.Rotate(v=radNdeg)
            y2 = 1 - newVec.x * newVec.x
            d = abs(abs(newVec.y**2) - abs(y2))
            print(f"add {i:3d}deg = {newVec} Δ={d:+.18f}")

    def TestSetAngle():
        print("------------ TestSetAngle ------------")
        vec = Vec2D(1, 1)
        for i in [0, 30, 45, 90, 180, 270, 340, -75, -90, -150, -350]:
            radNdeg = Rad.fromDeg(v=i)
            newVec = vec.SetAngle(radNdeg)
            d = abs(i - newVec.angle.toDegFloat())
            d = d if d < 360 else d % 360
            print(f"set {i:3d}deg = {newVec} Δ={d:+.18f}")

    def TestSetSize():
        print("------------ Test SetSize ------------")
        vec = Vec2D(1, 1)
        for i in [1, 2, 4, 9]:
            newVec = vec.SetNorm(i)
            d = abs(i - newVec.norm)
            print(f"resize {i:+.2f} = {newVec} Δ={d:+.18f}")

    TestCumulativeRotationError()
    TestRotationError()
    TestSetAngle()
    TestSetSize()

    pass


def main():
    _Vector2D_Test()
    pass


if __name__ == "__main__":
    main()
