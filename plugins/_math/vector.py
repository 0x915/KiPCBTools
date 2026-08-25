from __future__ import annotations

import math
from enum import IntEnum
from pathlib import Path
from typing import Literal, Self, Tuple

from _logging import Config, PrefixLogger

from .public import Num, format_num, TOLCF2, TOLCF3
from .radian import Rad

printlog = PrefixLogger(
    Path(__file__).name,
    sinks=[Config.ColorStdoutSink()],
    dyer=Config.default_dyer,
)

DEFAULT_TOL = 0.1e-4


class Vec2D:
    class ZeroVecError(Exception): ...

    def __init__(self, x: Num, y: Num):
        self.x: Num = x
        self.y: Num = y

    def clone(self) -> Vec2D:
        return Vec2D(self.x, self.y)

    def to_tuple(self) -> Tuple[Num, Num]:
        return self.x, self.y

    def set_value(self, x: Num, y: Num):
        self.x = x
        self.y = y

    def add_value(self, x: Num, y: Num) -> None:
        self.x += x
        self.y += y

    ## 覆盖内置方法

    PRINT_FLOAT_DIGIT_NUM = 6

    def __str__(self) -> str:
        DIG = self.PRINT_FLOAT_DIGIT_NUM
        return f"XY({format_num(self.x, DIG)},{format_num(self.y, DIG)})"

    def __repr__(self) -> str:
        DIG = self.PRINT_FLOAT_DIGIT_NUM
        return f"<{self.__class__.__name__}({format_num(self.x, DIG)},{format_num(self.y, DIG)}) At0x{id(self):X}>"
    
    def __eq__(self, v) -> bool:
        return (self.x == v.x) and (self.y == v.y)

    def __add__(self, v: Vec2D) -> Vec2D:
        return Vec2D(self.x + v.x, self.y + v.y)

    def __sub__(self, v: Vec2D) -> Vec2D:
        return Vec2D(self.x - v.x, self.y - v.y)

    def __mul__(self, v: Num) -> Vec2D:
        return Vec2D(self.x * v, self.y * v)

    def __truediv__(self, v: Num):
        return Vec2D(self.x / v, self.y / v)

    def __neg__(self) -> Vec2D:
        return Vec2D(-self.x, -self.y)

    def add(self, v: Self):
        self.x += v.x
        self.y += v.y

    def sub(self, v: Vec2D):
        self.x -= v.x
        self.y -= v.y

    def mul(self, v: Num):
        self.x *= v
        self.y *= v

    def div(self, v: Num):
        self.x /= v
        self.y /= v

    def neg(self):
        self.x = -self.x
        self.y = -self.y

    ## 向量特性

    def is_zero(self):
        return (self.x == 0) and (self.y == 0)

    def raise_zero(self):
        if self.is_zero():
            raise Vec2D.ZeroVecError
        return

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def set_norm(self, v: Num):
        self.raise_zero()
        self.mul(v / self.norm())

    def sin(self, ref: Vec2D | None = None) -> float:
        # SinV=|a×b|/(|a|·|b|)
        self.raise_zero()
        if ref is None:
            return self.y / self.norm()
        ref.raise_zero()
        return Vec2D.cross(ref, self) / (self.norm() * ref.norm())

    def cos(self, ref: Vec2D | None = None) -> float:
        # CosV=(a·b)/(|a|·|b|)
        self.raise_zero()
        if ref is None:
            return self.x / self.norm()
        ref.raise_zero()
        return Vec2D.dot(ref, self) / (self.norm() * ref.norm())

    def angle(self, ref: Vec2D | None = None) -> Rad:
        self.raise_zero()
        if ref is None:
            a = math.atan2(self.y, self.x)
        else:
            ref.raise_zero()
            a = math.atan2(Vec2D.cross(ref, self), Vec2D.dot(ref, self))
        rad = Rad(a)
        rad.to_positive()
        return rad

    def set_angle(self, v: Rad) -> None:
        self.raise_zero()
        self.rotate(v - self.angle())

    def rotate(self, v: Rad) -> None:
        self.raise_zero()
        cosV = v.cos()
        sinV = v.sin()
        x = self.x * cosV - self.y * sinV
        y = self.x * sinV + self.y * cosV
        self.set_value(x, y)

    ## 静态工具集

    @staticmethod
    def dot(ref: Vec2D, vec: Vec2D):
        """点积：零值正交"""
        return (ref.x * vec.x) + (ref.y * vec.y)

    @staticmethod
    def cross(ref: Vec2D, vec: Vec2D):
        """叉积：零值平行"""
        return (ref.x * vec.y) - (vec.x * ref.y)

    @staticmethod
    def cosine_similarity(a: Vec2D, b: Vec2D) -> float:
        """正弦相似度：零值正交，正值方向(+0,90)(270,-0)，负值方向(90,180)(180,270)"""
        return a.cos(ref=b)

    @staticmethod
    def sine_similarity(a: Vec2D, b: Vec2D) -> float:
        """正弦相似度：零值平行，正值方向(+0,180)，负值方向(180,-0)"""
        return a.sin(ref=b)

    @staticmethod
    def is_orthogonal(u: Vec2D, v: Vec2D, threshold=DEFAULT_TOL) -> bool:
        """判断正交"""
        return abs(u.cos(ref=v)) <= abs(threshold)

    @staticmethod
    def get_orthogonal(u: Vec2D, v: Vec2D, threshold=DEFAULT_TOL) -> Literal[0, 1, -1]:
        """获取正交关系：返回0=不正交，返回-1=负角度侧，返回+1=正角度侧"""
        u.raise_zero()
        v.raise_zero()
        if Vec2D.is_orthogonal(u, v, threshold) is False:
            return 0
        if Vec2D.cross(u, v) > 0:
            return 1
        return -1

    @staticmethod
    def is_parallel(u: Vec2D, v: Vec2D, tol=DEFAULT_TOL) -> bool:
        """判断平行"""
        return abs(u.sin(ref=v)) <= abs(tol)

    @staticmethod
    def get_parallel(u: Vec2D, v: Vec2D, tol=DEFAULT_TOL) -> Literal[0, 1, -1]:
        """获取平行关系：返回0=不平行，返回-1=负角度侧，返回+1=正角度侧"""
        u.raise_zero()
        v.raise_zero()
        if not Vec2D.is_parallel(u, v, tol):
            return 0
        if Vec2D.dot(u, v) > 0:
            return 1
        return -1


class Line2D:
    def __init__(self, start: Vec2D, end: Vec2D) -> None:
        self.start_pt = start
        self.end_pt = end

    def __str__(self) -> str:
        return f"Line({self.start_pt},{self.end_pt})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} start={self.start_pt} end={self.end_pt} At0x{id(self):X}>"
    
    def get_vec(self):
        return self.end_pt - self.start_pt

    class CollinearRelation(IntEnum):
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
    def get_collinear(
        a: Line2D,
        b: Line2D,
        clearance_tol: float = DEFAULT_TOL,
        xy_tol: float = DEFAULT_TOL,
    ) -> CollinearRelation | None:
        """获取共线关系(误差校正)"""
        vec_a = a.get_vec()
        vec_b = b.get_vec()
        vec_a.raise_zero()
        vec_b.raise_zero()
        # 检查平行(距离差判据)
        if Line2D.is_parallel(a, b, clearance_tol) is False:
            return None
        # 检查间距
        clearance = Line2D.get_point_distance(a, b.end_pt)
        if clearance >= clearance_tol:
            return None

        # 选择u和v中最大的向量
        lu = vec_a.norm()
        lv = vec_b.norm()
        if lu >= lv:
            long_line, short_line = a, b
            size_l, size_s = lu, lv
        else:
            long_line, short_line = b, a
            size_l, size_s = lv, lu

        # 投影 = 短线段端点在长线段方向上的带符号投影
        """
        # s | -->     | -->   | -->   | -->   | -----> |  -->   |   --> |    ---> |     ---> |        --> |
        # l |     --> |   --> |  -->  | ----> | -----> | -----> | ----> | ---->   | ---->    | ---->      |
        # w |  <--    |       |  ->   | -->   | -----> | --->   | ----> | ------> | -------> | ---------> |
        #   |   -w    |  w=0  | w<s<l | w=s<l |  w=s=l | s<w<l  | s<w=l | l>w>s+l |   w=s+l  |   w>s+l    |
        #   | NOT(S)  | CC(S) | OL(S) | CT(S) |   EQ   |   CT   | CT(E) | OL(E)   |   CC(E)  |   NOT(S)   |
        """
        if Vec2D.dot(vec_a, vec_b) > 0:
            p = Line2D.get_projection_length(long_line, short_line.end_pt)
        else:
            p = Line2D.get_projection_length(long_line, short_line.start_pt)

        # p = round(p,float_digs)
        # size_l = round(size_l,float_digs)
        # size_s = round(size_s,float_digs)
        size_sl = size_l + size_s

        ## NOT(S) 不包含(在负方向)    投影 负值
        # p < 0:
        if TOLCF2(p, "<", 0, xy_tol):
            return Line2D.CollinearRelation.BeforeStart
        ## CC(S) 不包含(共长向量起点) 投影 零值
        # p == 0
        elif TOLCF2(p, "==", 0, xy_tol):
            return Line2D.CollinearRelation.ConnectStart

        ## OL(S) 包含部分(在起点上)   投影<短向量
        # p < size_s < size_l
        elif TOLCF3(p, "<", size_s, "<", size_l, xy_tol):
            return Line2D.CollinearRelation.OverlapStart
        ## CT(S) 包含(在起点上)      投影=短向量
        # p == size_s < size_l
        elif TOLCF3(p, "==", size_s, "<", size_l, xy_tol):
            return Line2D.CollinearRelation.ContainStart

        ## EQ 特殊 包含(在两端点上)   投影=短向量=长向量
        # p == size_s == size_l
        elif TOLCF3(p, "==", size_s, "==", size_l, xy_tol):
            return Line2D.CollinearRelation.ContainEqual
        ## CT 内部 包含(不在端点上)   投影>短向量
        # size_s < p < size_l
        elif TOLCF3(size_s, "<", p, "<", size_l, xy_tol):
            return Line2D.CollinearRelation.ContainInside

        ## CT(E) 包含(在终点上)      投影=长向量
        # size_s < p == size_l
        elif TOLCF3(size_s, "<", p, "==", size_l, xy_tol):
            return Line2D.CollinearRelation.ContainEnd
        ## OL(E) 包含(在终点上)      投影<小于长向量加短向量
        # size_l < p < size_sl
        elif TOLCF3(size_l, "<", p, "<", size_sl, xy_tol):
            return Line2D.CollinearRelation.OverlapEnd

        ## CC(E) 不包含(共长向量终点) 投影=长向量加短向量
        # size_l < p == size_sl
        elif TOLCF3(size_l, "<", p, "==", size_sl, xy_tol):
            return Line2D.CollinearRelation.ConnectEnd
        ## NOT(S) 不包含(在正方向)    投影>长向量加短向量
        # size_l < size_sl < p
        elif TOLCF3(size_l, "<", size_sl, "<", p, xy_tol):
            return Line2D.CollinearRelation.AfterEnd

        raise Exception("不该出现的错误")

    @staticmethod
    def make_orthogonal(line: Line2D, size: Num) -> Vec2D:
        """生成正交向量"""
        u = line.get_vec()
        u.raise_zero()
        return Vec2D(size * u.sin(), size * u.cos())

    @staticmethod
    def make_parallel(line: Line2D, clearance: Num) -> Line2D:
        """生成平行线段"""
        r90 = Line2D.make_orthogonal(line, clearance)
        r90.x = -r90.x
        return Line2D(line.start_pt + r90, line.end_pt + r90)

    @staticmethod
    def get_point_distance(line: Line2D, pt: Vec2D) -> float:
        """
        计算 pt 到 line 的垂直距离 \n
        `+f` 点在line正角度侧上，距离=f \n
        `=0` 点在line上 \n
        `-f` 点在line负角度侧上，距离=f \n
        """
        u = line.get_vec()
        u.raise_zero()
        w = pt - line.start_pt
        return 0 if w.is_zero() else (w.norm() * w.sin(ref=u))

    @staticmethod
    def is_parallel(a: Line2D, b: Line2D, tol: Num) -> bool:
        """
        判断平行(距离差判据) \n
        d1/d2 = b 两端点到直线 a 的有符号垂直距离  |d1-d2| = |b|·|sinθ| \n
        tol 为距离量纲(无默认值 按调用方坐标单位提供): 两端点垂直距离差的允许值 与段长无关
        """
        d1 = Line2D.get_point_distance(a, b.start_pt)
        d2 = Line2D.get_point_distance(a, b.end_pt)
        return abs(d1 - d2) <= abs(tol)

    @staticmethod
    def is_orthogonal(a: Line2D, b: Line2D, tol: Num) -> bool:
        """
        判断正交(距离差判据) \n
        正交时 |d1-d2| = |b| 即两端点垂直距离差等于 b 的线长 \n
        tol 为距离量纲(无默认值 按调用方坐标单位提供): 与线长的允许偏差
        """
        d1 = Line2D.get_point_distance(a, b.start_pt)
        d2 = Line2D.get_point_distance(a, b.end_pt)
        blen = b.get_vec().norm()
        return abs(abs(d1 - d2) - blen) <= abs(tol)

    @staticmethod
    def get_parallel_clearance(a: Line2D, b: Line2D, threshold: Num) -> float | None:
        """
        计算 a 与 b 的平行间距 \n
        `+f` b 在 a 正角度侧，距离=f \n
        `=0` b 与 a 共线 \n
        `-f` b 在 a 负角度侧，距离=f \n
        `None` b 和 a 不平行(两端点垂直距离差超出 threshold) \n
        """
        a.get_vec().raise_zero()
        b.get_vec().raise_zero()
        if Line2D.is_parallel(a, b, threshold) is False:
            return None
        return Line2D.get_point_distance(a, b.end_pt)

    @staticmethod
    def get_projection_length(line: Line2D, pt: Vec2D) -> float:
        """
        计算 pt 在 line 起点上的投影距离 \n
        `+f` 投影在line正方向上，距离=f \n
        `=0` 垂直方向上 \n
        `-f` 投影在line负方向上，距离=f \n
        """
        v = line.get_vec()
        v.raise_zero()
        w = pt - line.start_pt
        return 0 if w.is_zero() else (w.norm() * w.cos(ref=v))

    @staticmethod
    def get_linear_junction(a: Line2D, b: Line2D) -> Vec2D | None:
        a.get_vec().raise_zero()
        b.get_vec().raise_zero()
        x1, y1 = a.start_pt.to_tuple()
        x2, y2 = a.end_pt.to_tuple()
        x3, y3 = b.start_pt.to_tuple()
        x4, y4 = b.end_pt.to_tuple()

        # 共点检查
        us_vs = x1 == x3 and y1 == y3
        us_ve = x1 == x4 and y1 == y4
        ue_ve = x2 == x4 and y2 == y4
        ue_vs = x2 == x3 and y2 == y3

        # 正向同线(+) u ·---->· v
        # 反向同线(-) u ·<--->· v
        if (us_vs and ue_ve) or (us_ve and ue_vs):
            return None

        # 共U起点(-) u <----·----> v
        # 共U起点(+) u <----·<---- v
        elif us_vs or us_ve:
            return Vec2D(x1, y1)

        # 共U终点(-) u ---->·<---- v
        # 共U终点(+) u ---->·----> v
        elif ue_ve or ue_vs:
            return Vec2D(x2, y2)

        x_up = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        x_dn = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if x_dn == 0:
            return None
        y_up = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        return Vec2D(x_up / x_dn, y_up / x_dn)

    @staticmethod
    def set_junction_connect(u: Line2D, ref: Line2D, set_end: bool = True) -> bool:
        junction = Line2D.get_linear_junction(u, ref)
        if junction is None:
            return False

        if junction == u.start_pt:
            return True
        if junction == u.end_pt:
            return True

        if set_end is True:
            u.end_pt = junction
        else:
            u.start_pt = junction

        return True

    @staticmethod
    def test():
        pt1 = Vec2D(+125416610, +95966632)
        pt2 = Vec2D(+125799143, +94539003)
        printlog.info(f"Point1 = {pt1}")
        printlog.info(f"Point2 = {pt2}")
        line1 = Line2D(pt1, pt2)
        printlog.info(f"Line1 = {line1}")

        pt3 = Vec2D(+125146152, +95894162)
        pt4 = Vec2D(+125498666, +94578561)
        printlog.info(f"Point3 = {pt3}")
        printlog.info(f"Point4 = {pt4}")
        line2 = Line2D(pt3, pt4)
        printlog.info(f"Line2 = {line2}")

        printlog.info(f"get_parallel_clearance = {Line2D.get_parallel_clearance(line1, line2, 0.000001)}")


def __test_line2d(): ...


if __name__ == "__main__":
    Line2D.test()
