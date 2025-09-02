import math
import copy
from enum import IntEnum
from typing import Tuple

from MathLib.Define import Num, NumOrNone, StrNum, EQ, TOLCF2, TOLCF3
from MathLib.Radian import Rad

_DEFAULT_TOL = 0.000001


class Vec2D:
    ## 构造函数

    def __init__(self, x: Num, y: Num):
        assert isinstance(x, Num), f"类{self.__class__.__name__}构造函数 x 输入类型 {type(x)} 错误"
        assert isinstance(x, Num), f"类{self.__class__.__name__}构造函数 y 输入类型 {type(y)} 错误"
        self.x: Num = x
        self.y: Num = y
        self.offset: Vec2D | None = None

    @staticmethod
    def fromXY(end: "Vec2D") -> "Vec2D":
        return Vec2D(end.x, end.y)

    @staticmethod
    def fromXYXY(start: "Vec2D", end: "Vec2D"):
        u = Vec2D(0, 0)
        u.SetEnd(end.x, end.y)
        u.SetStart(start.x, start.y)
        return u

    @staticmethod
    def fromOffset(offset: "Vec2D", vec: "Vec2D"):
        u = Vec2D(0, 0)
        u.offset = Vec2D(offset.x, offset.y)
        u.Set(vec.x, vec.y)
        return u

    def Clone(self) -> "Vec2D":
        return copy.deepcopy(self)

    ## 向量端点

    def GetTuple(self) -> Tuple[Num, Num]:
        """获取 向量值"""
        return self.x, self.y

    def Get(self) -> "Vec2D":
        return Vec2D(self.x, self.y)

    def GetStart(self) -> "Vec2D":
        """获取 实际起点坐标"""
        if self.offset is None:
            return Vec2D(0, 0)
        return self.offset.GetEnd()

    def GetEnd(self):
        """获取 实际终点坐标"""
        if self.offset is None:
            return Vec2D(self.x, self.y)
        return self.GetStart() + self

    def Set(self, x: Num, y: Num):
        """设置 向量值"""
        self.x = x
        self.y = y

    def MoveStart(self, x: Num, y: Num):
        """移动实际起点 [改变终点+锁定向量大小]"""
        if self.offset is None:
            self.offset = Vec2D(x, y)
        else:
            self.offset.SetEnd(x, y)

    def SetStart(self, x: Num, y: Num):
        """设置实际起点 [锁定终点+改变向量大小]"""
        end = self.GetEnd()
        self.Set(end.x - x, end.y - y)
        if self.offset is None:
            self.offset = Vec2D(x, y)
        else:
            self.offset.SetEnd(x, y)

    def MoveEnd(self, x: Num, y: Num):
        """移动实际终点 [改变起点+锁定向量大小]"""
        self.SetStart(x - self.x, y - self.y)
        # Start=End-Vec

    def SetEnd(self, x: Num, y: Num):
        """设置实际终点 [锁定起点+改变向量大小]"""
        start = self.GetStart()
        self.Set(x - start.x, y - start.y)
        # Vec=End-Start

    ## 向量特性

    def isZero(self):
        """检查 零向量"""
        return (self.x == 0) and (self.y == 0)

    def norm(self) -> float:
        """获取 模长"""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def SetNorm(self, v: Num):
        """设置 模长 交互零向量会抛出异常"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        self.Mul(v / self.norm())

    def sin(self, ref: "Vec2D|None" = None) -> float:
        """设置 正弦 交互零向量会抛出异常"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        if ref is None:
            return self.y / self.norm()
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        return Vec2D.Cross(ref, self) / (self.norm() * ref.norm())

    def cos(self, ref: "Vec2D|None" = None) -> float:
        """获取 余弦 CosV = u•v / |u|•|v|"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        if ref is None:
            return self.x / self.norm()
        assert not ref.isZero(), f"INPUT {ref} is ZeroVec"
        return Vec2D.Dot(ref, self) / (self.norm() * ref.norm())

    def angle(self, ref: "Vec2D|None" = None) -> Rad:
        """获取 角度 交互零向量会抛出异常"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        if ref is None:
            a = math.atan2(self.y, self.x)
        else:
            assert not ref.isZero(), f"INPUT {ref} is ZeroVec"
            a = math.atan2(Vec2D.Cross(ref, self), Vec2D.Dot(ref, self))
        rad = Rad(a)
        rad.toPositive()
        return rad

    def SetAngle(self, v: Rad) -> None:
        """设置 角度 交互零向量会抛出异常"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        self.Rotate(v - self.angle())

    def Rotate(self, v: Rad) -> None:
        """设置 角度 交互零向量会抛出异常"""
        assert not self.isZero(), f"SELF {self} is ZeroVec"
        cosV = v.cos()
        sinV = v.sin()
        x = self.x * cosV - self.y * sinV
        y = self.x * sinV + self.y * cosV
        self.x = x
        self.y = y

    ## 静态工具集

    @staticmethod
    def Dot(ref: "Vec2D", vec: "Vec2D"):
        """内积/点乘"""
        # return u.norm() * v.norm() * u.cos(v)
        return (ref.x * vec.x) + (ref.y * vec.y)

    @staticmethod
    def Cross(ref: "Vec2D", vec: "Vec2D"):
        """外积/叉乘"""
        return (ref.x * vec.y) - (vec.x * ref.y)

    @staticmethod
    def GetIncludedAngle(u: "Vec2D", v: "Vec2D"):
        """角度差"""
        return v.angle() - u.angle()

    @staticmethod
    def CosineSimilarity(a: "Vec2D", b: "Vec2D") -> float:
        """正弦相似度 零值正交 正值方向(+0,90)(270,-0) 负值方向(90,180)(180,270)"""
        return a.cos(ref=b)

    @staticmethod
    def SineSimilarity(a: "Vec2D", b: "Vec2D") -> float:
        """正弦相似度：零值平行 正值方向(+0,180) 负值方向(180,-0)"""
        return a.sin(ref=b)

    class RET:
        class Parallel(IntEnum):
            NOT = 0
            Positive = 1
            Negative = -1

        class Orthogonal(IntEnum):
            NOT = 0
            Positive = 1
            Negative = -1

        class Relation(IntEnum):
            FlagNot = 0b1 << 16  # 错误
            FlagContain = 0b1 << 17  # 包含
            FlagConnect = 0b1 << 18  # 共点
            FlagStartPoint = 0b1 << 19  # 起点
            FlagEndPoint = 0b1 << 20  # 终点

            NotParallel = int(FlagNot | 11)  # 不平行
            NotCollinear = int(FlagNot | 12)  # 不共线

            BeforeStart = int(FlagStartPoint | 1)  # 在负方向 不包含
            ConnectStart = int(FlagStartPoint | FlagConnect | 2)  # 共起点   不包含
            OverlapStart = int(FlagStartPoint | FlagContain | 3)  # 共起点 部分包含
            ContainStart = int(FlagStartPoint | FlagContain | FlagConnect | 4)  # 共起点     包含
            ContainEqual = int(FlagStartPoint | FlagEndPoint | FlagContain | 5)  # 特殊   相互包含
            ContainInside = int(FlagContain | 6)  # 内部   完全包含
            ContainEnd = int(FlagEndPoint | FlagContain | FlagConnect | 7)  # 共终点     包含
            OverlapEnd = int(FlagEndPoint | FlagContain | 8)  # 共终点 部分包含
            ConnectEnd = int(FlagEndPoint | FlagConnect | 9)  # 共终点   不包含
            AfterEnd = int(FlagEndPoint | 10)  # 在正方向 不包含

    @staticmethod
    def isOrthogonal(u: "Vec2D", v: "Vec2D", tol=_DEFAULT_TOL) -> bool:
        """判断正交"""
        return True if abs(Vec2D.Dot(u, v)) <= abs(tol) else False

    @staticmethod
    def GetOrthogonal(u: "Vec2D", v: "Vec2D", tol=_DEFAULT_TOL) -> "Vec2D.RET.Orthogonal":
        """获取正交关系"""
        assert not u.isZero(), "未定义行为:零向量?正交?"
        assert not v.isZero(), "未定义行为:零向量?正交?"
        if not Vec2D.isOrthogonal(u, v, tol):
            return Vec2D.RET.Orthogonal.NOT
        # 叉积为正 正角度侧
        if Vec2D.Cross(u, v) > 0:
            return Vec2D.RET.Orthogonal.Positive
        return Vec2D.RET.Orthogonal.Negative

    @staticmethod
    def isParallel(u: "Vec2D", v: "Vec2D", tol=_DEFAULT_TOL) -> bool:
        """判断平行"""
        return abs(Vec2D.Cross(u, v)) <= abs(tol)

    @staticmethod
    def GetParallel(u: "Vec2D", v: "Vec2D", tol=_DEFAULT_TOL) -> "Vec2D.RET.Parallel":
        """获取平行关系"""
        assert not u.isZero(), "未定义行为:零向量?平行?"
        assert not v.isZero(), "未定义行为:零向量?平行?"
        if not Vec2D.isParallel(u, v, tol):
            return Vec2D.RET.Parallel.NOT
        # 点积为正 同方向
        if Vec2D.Dot(u, v) > 0:
            return Vec2D.RET.Parallel.Positive
        return Vec2D.RET.Parallel.Negative

    @staticmethod
    def GetCollinear(u: "Vec2D", v: "Vec2D", clearance_tol: float = _DEFAULT_TOL, xy_tol: float = _DEFAULT_TOL) -> "Vec2D.RET.Relation":
        """获取共线关系(误差校正)"""
        assert not u.isZero(), "未定义行为:零向量?共线?"
        assert not v.isZero(), "未定义行为:零向量?共线?"
        # 检查平行
        parallel = Vec2D.GetParallel(u, v)
        if parallel == Vec2D.RET.Parallel.NOT:
            return Vec2D.RET.Relation.NotParallel
        # 检查间距
        clearance = Vec2D.GetPointDistance(u, v.GetEnd())
        if clearance >= clearance_tol:
            return Vec2D.RET.Relation.NotCollinear

        # 选择u和v中最大的向量
        lu = u.norm()
        lv = v.norm()
        if lu >= lv:
            vlong, vshort = u, v
            size_l, size_s = lu, lv
        else:
            vlong, vshort = v, u
            size_l, size_s = lv, lu

        # 投影向量w = 同向[v起点->u终点] 反向[v起点->u起点]
        """
        # s | -->     | -->   | -->   | -->   | -----> |  -->   |   --> |    ---> |     ---> |        --> |
        # l |     --> |   --> |  -->  | ----> | -----> | -----> | ----> | ---->   | ---->    | ---->      |
        # w |  <--    |       |  ->   | -->   | -----> | --->   | ----> | ------> | -------> | ---------> |
        #   |   -w    |  w=0  | w<s<l | w=s<l |  w=s=l | s<w<l  | s<w=l | l>w>s+l |   w=s+l  |   w>s+l    |
        #   | NOT(S)  | CC(S) | OL(S) | CT(S) |   EQ   |   CT   | CT(E) | OL(E)   |   CC(E)  |   NOT(S)   |
        """
        if parallel == Vec2D.RET.Parallel.Positive:
            w = vshort.GetEnd() - vlong.GetStart()
        else:
            w = vshort.GetStart() - vlong.GetStart()
        p = Vec2D.GetProjectionSize(w, vlong)

        # p = round(p,float_digs)
        # size_l = round(size_l,float_digs)
        # size_s = round(size_s,float_digs)
        size_sl = size_l + size_s

        ## NOT(S) 不包含(在负方向)    投影 负值
        # p < 0:
        if TOLCF2(p, "<", 0, xy_tol):
            return Vec2D.RET.Relation.BeforeStart
        ## CC(S) 不包含(共长向量起点) 投影 零值
        # p == 0
        elif TOLCF2(p, "==", 0, xy_tol):
            return Vec2D.RET.Relation.ConnectStart

        ## OL(S) 包含部分(在起点上)   投影<短向量
        # p < size_s < size_l
        elif TOLCF3(p, "<", size_s, "<", size_l, xy_tol):
            return Vec2D.RET.Relation.OverlapStart
        ## CT(S) 包含(在起点上)      投影=短向量
        # p == size_s < size_l
        elif TOLCF3(p, "==", size_s, "<", size_l, xy_tol):
            return Vec2D.RET.Relation.ContainStart

        ## EQ 特殊 包含(在两端点上)   投影=短向量=长向量
        # p == size_s == size_l
        elif TOLCF3(p, "==", size_s, "==", size_l, xy_tol):
            return Vec2D.RET.Relation.ContainEqual
        ## CT 内部 包含(不在端点上)   投影>短向量
        # size_s < p < size_l
        elif TOLCF3(size_s, "<", p, "<", size_l, xy_tol):
            return Vec2D.RET.Relation.ContainInside

        ## CT(E) 包含(在终点上)      投影=长向量
        # size_s < p == size_l
        elif TOLCF3(size_s, "<", p, "==", size_l, xy_tol):
            return Vec2D.RET.Relation.ContainEnd
        ## OL(E) 包含(在终点上)      投影<小于长向量加短向量
        # size_l < p < size_sl
        elif TOLCF3(size_l, "<", p, "<", size_sl, xy_tol):
            return Vec2D.RET.Relation.OverlapEnd

        ## CC(E) 不包含(共长向量终点) 投影=长向量加短向量
        # size_l < p == size_sl
        elif TOLCF3(size_l, "<", p, "==", size_sl, xy_tol):
            return Vec2D.RET.Relation.ConnectEnd
        ## NOT(S) 不包含(在正方向)    投影>长向量加短向量
        # size_l < size_sl < p
        elif TOLCF3(size_l, "<", size_sl, "<", p, xy_tol):
            return Vec2D.RET.Relation.AfterEnd

        raise Exception("不该出现的错误")

    @staticmethod
    def MakeOrthogonal(u: "Vec2D", size: Num):
        """生成正交向量"""
        assert size != 0, "新向量的模长不能为零"
        assert not u.isZero(), "未定义行为:零向量?正交?"
        return Vec2D(size * u.sin(), size * u.cos())

    @staticmethod
    def MakeParallelLine(u: "Vec2D", clearance: Num):
        """生成平行向量"""
        assert clearance != 0, "平行间距不能为零"
        assert not u.isZero(), "未定义行为:零向量?平行?"
        r90 = Vec2D.MakeOrthogonal(u, clearance)
        r90.x = -r90.x
        return Vec2D.fromXYXY(u.GetStart() + r90, u.GetEnd() + r90)

    @staticmethod
    def GetPointDistance(u: "Vec2D", pt: "Vec2D") -> float:
        """
        计算 pt 到 u 距离 \n
        +f 点在u正角度侧上的距离 \n
        =0 点和u共线 \n
        -f 点在u负角度侧上的距离 \n
        """
        assert not u.isZero(), "未定义行为:点与零向量的正交距离"
        w = pt - u.GetStart()
        return 0 if w.isZero() else (w.norm() * w.sin(ref=u))

    @staticmethod
    def GetParallelClearance(u: "Vec2D", v: "Vec2D", tol=_DEFAULT_TOL) -> float | None:
        """
        计算 u 与 v 的平行间距 \n
        +f 投影在v正方向上 \n
        =0 垂直(不存在投影) \n
        -f 投影在v负方向上 \n
        """
        assert not u.isZero(), "未定义行为:零向量?平行?"
        assert not v.isZero(), "未定义行为:零向量?平行?"
        if Vec2D.isParallel(u, v, tol):
            return Vec2D.GetPointDistance(u, v.GetEnd())
        return None

    @staticmethod
    def GetProjectionSize(u: "Vec2D", v: "Vec2D") -> float:
        """
        计算 u 在 v 的投影距离 \n
        +f 投影在v正方向上 \n
        =0 垂直(不存在投影) \n
        -f 投影在v负方向上 \n
        """
        assert not v.isZero(), "未定义行为:投影在零向量上?"
        return 0 if u.isZero() else (u.norm() * u.cos(ref=v))

    @staticmethod
    def GetLinearJunction(u: "Vec2D", v: "Vec2D") -> "Vec2D| None":
        assert not u.isZero(), "未定义行为:零向量?交点?"
        assert not v.isZero(), "未定义行为:零向量?交点?"
        x1, y1 = u.GetStart().GetTuple()
        x2, y2 = u.GetEnd().GetTuple()
        x3, y3 = v.GetStart().GetTuple()
        x4, y4 = v.GetEnd().GetTuple()

        # 共点检查
        is_us_vs = x1 == x3 and y1 == y3
        is_us_ve = x1 == x4 and y1 == y4
        is_ue_ve = x2 == x4 and y2 == y4
        is_ue_vs = x2 == x3 and y2 == y3

        # 正向同线(+) u ·---->· v
        # 反向同线(-) u ·<--->· v
        if (is_us_vs and is_ue_ve) or (is_us_ve and is_ue_vs):
            raise Exception("未定义行为:对两个等长重叠的向量求交点")
        # 共U起点(-) u <----·----> v
        # 共U起点(+) u <----·<---- v
        elif is_us_vs or is_us_ve:
            return Vec2D(x1, y1)
        # 共U终点(-) u ---->·<---- v
        # 共U终点(+) u ---->·----> v
        elif is_ue_ve or is_ue_vs:
            return Vec2D(x2, y2)

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
    def JunctionConnect(u: "Vec2D", ref: "Vec2D", set_end: bool = True) -> "bool":
        junction = Vec2D.GetLinearJunction(u, ref)
        if junction is None:
            return False

        if junction == u.GetStart():
            return True
        if junction == u.GetEnd():
            return True
        if set_end:
            u.SetEnd(junction.x, junction.y)
        else:
            u.SetStart(junction.x, junction.y)

        return True

    ## 覆盖内置方法

    PRINT_FLOAT_DIGIT_NUM = 6

    def __str__(self) -> str:
        DIG = self.PRINT_FLOAT_DIGIT_NUM
        start = self.GetStart()
        return f"<{self.__class__.__name__} XY({StrNum(self.x, DIG)}, {StrNum(self.y, DIG)}) AT({StrNum(start.x, DIG)},{StrNum(start.y, DIG)}) 0x{id(self):X}>"

    def __eq__(self, v) -> bool:
        return (self.x == v.x) and (self.y == v.y)

    def __add__(self, v: "Vec2D") -> "Vec2D":
        return Vec2D(self.x + v.x, self.y + v.y)

    def __sub__(self, v: "Vec2D") -> "Vec2D":
        return Vec2D(self.x - v.x, self.y - v.y)

    def __mul__(self, v: Num) -> "Vec2D":
        return Vec2D(self.x * v, self.y * v)

    def __truediv__(self, v: Num):
        return Vec2D(self.x / v, self.y / v)

    def __neg__(self) -> "Vec2D":
        return Vec2D(-self.x, -self.y)

    def formatPoint(self) -> str:
        DIG = self.PRINT_FLOAT_DIGIT_NUM
        start = self.GetStart()
        end = self.GetEnd()
        return (
            f"<{self.__class__.__name__} Start({StrNum(start.x, DIG)}, {StrNum(start.y, DIG)}) End({StrNum(end.x, DIG)},{StrNum(end.y, DIG)}) 0x{id(self):X}>"
        )

    def AddXY(self, x: Num, y: Num) -> None:
        # fmt:off
        if x:self.x+=x
        if x:self.y+=y
        # fmt:on

    def AddVec2D(self, v: "Vec2D"):
        self.x += v.x
        self.y += v.y

    def Sub(self, v: "Vec2D"):
        self.x -= v.x
        self.y -= v.y

    def Mul(self, v: Num):
        self.x *= v
        self.y *= v

    def Div(self, v: Num):
        self.x /= v
        self.y /= v

    def Neg(self):
        self.x = -self.x
        self.y = -self.y
