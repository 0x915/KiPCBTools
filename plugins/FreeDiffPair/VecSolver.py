from typing import Callable, Generic, List, Protocol, Tuple, TypeVar

from .TrackExport import (
    TrackPolyline2i,
    TrackPoint2i,
    TrackLike,
    ExportInfo,
    ExportInfo_Result,
    ExportPoint,
    ExportLine,
)


from . import logger as printlog


from _math import Vec2D, Line2D

KIUNIT_CLEARANCE_TOL:int = 10000

T = TypeVar("T")


class _CursorList(Generic[T]):
    """带游标的元素列表基类"""

    def __init__(self) -> None:
        self._i: int = 0
        self._list: List[T] = []

    def GetList(self):
        return self._list

    def Append(self, item: T):
        self._list.append(item)

    def Count(self):
        return len(self._list)

    def pMoveStart(self):
        self._i = 0

    def pMoveEnd(self):
        self._i = self.Count() - 1

    def pMoveNext(self):
        if self._i >= self.Count() - 1:
            return False
        self._i += 1
        return True

    def pMovePrev(self):
        if self._i <= 0:
            return False
        self._i -= 1
        return True

    def pGet(self):
        return self._i

    def pSet(self, p: int):
        if p >= self.Count():
            raise OverflowError
        if p <= 0:
            raise ValueError
        self._i = p

    def GetCurrent(self):
        return self._list[self._i]

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} 0x{id(self):X} n:{self.Count()}>"


class VecList2D(_CursorList[Vec2D]):
    """Vec2D 端点坐标列表"""


class LineList2D(_CursorList[Line2D]):
    """Line2D 折线段列表"""


def MakeVec2D(xy1xy2: Tuple[TrackPoint2i, TrackPoint2i]) -> Line2D:
    ptStart, ptEnd = xy1xy2
    return Line2D(Vec2D(ptStart.x, ptStart.y), Vec2D(ptEnd.x, ptEnd.y))


def PolylineToVecList(pl: TrackPolyline2i) -> LineList2D:
    mvec = LineList2D()

    printlog.info("")
    printlog.info(f"{PolylineToVecList.__name__}():")

    # 移动线段指针到开头
    pl.pMoveStart()

    printlog.info("折线向量化:")
    loop_break = pl.GetPointCount() + 2
    while True:
        loop_break -= 1
        assert loop_break > 0, "循环保护(构建向量时超出最大循环次数)"

        line = pl.GetCurrent()

        # 设计上不该出现这种情况 用于修正静态类型检查(排除None类型)
        assert line is not None, f"失败(只有{pl.GetPointCount()}个端点的非法折线)"

        # 构建线段向量 加入集合向量
        vec = MakeVec2D(line)
        if vec.get_vec().is_zero():
            printlog.info(f"  -跳过零长线段 ({vec.start_pt.x:+},{vec.start_pt.y:+})")
            if pl.pMoveNext() is False:
                break
            continue

        mvec.Append(vec)
        printlog.info(f"  +向量{mvec.Count()} {vec.start_pt}->{vec.end_pt}")

        if pl.pMoveNext() is False:
            break

    return mvec


def CheckPairPolar(referVec: Line2D, diffPt2: Tuple[TrackPoint2i, TrackPoint2i], clearance_tol=KIUNIT_CLEARANCE_TOL):
    printlog.info("")
    printlog.info(f"{CheckPairPolar.__name__}():")

    diffVec = MakeVec2D(diffPt2)
    diffVec.get_vec().raise_zero()
    printlog.info("检查输入差分对:")
    printlog.info(f"  参考 {referVec.start_pt} -> {referVec.end_pt}")
    printlog.info(f"  差分 {diffVec.start_pt} -> {diffVec.end_pt}")

    # 平行判定: 距离差判据 两端点到参考线的垂直距离差在容差内
    assert Line2D.is_parallel(referVec, diffVec, clearance_tol), "错误(起点差分对不平行)"

    # 方向判定: 同向+1 反向-1 决定差分线所在侧向
    ret = 1 if Vec2D.dot(diffVec.get_vec(), referVec.get_vec()) > 0 else -1
    if ret == -1:
        diffVec = MakeVec2D((diffPt2[1], diffPt2[0]))

    # 夹角精度: referVec 相对 diffVec 的有向角
    included = referVec.get_vec().angle(ref=diffVec.get_vec())
    printlog.info(f"  精度 {included.GetDeg():+.12f}(deg)")
    # 平行间距: 以 referVec 为基准 符号为 diffVec 终点在 referVec 的侧向
    # 使用与平行判定相同的距离容差 保证两处判定口径一致
    pair_distance = Line2D.get_parallel_clearance(referVec, diffVec, clearance_tol)
    printlog.debug(f"referVec = {referVec}")
    printlog.debug(f"diffVec = {diffVec}")
    assert pair_distance is not None, "错误(无法计算差分对间距)"
    printlog.info(f"  间距 {pair_distance:+}(unit)")

    assert pair_distance != 0, "错误(差分对(头)间距为零)"

    return ret, pair_distance


def GenerateNewPointList(
    sVecList: LineList2D,
    vec_distance: int,
) -> VecList2D:
    printlog.info("")
    printlog.info(f"{GenerateNewPointList.__name__}():")

    # 从单端线段表逐段构造指定距离的平行线段 边构造边修正交点
    # 平行线的偏置侧由 vec_distance 符号决定(MakeParallelLine 内部处理)
    points: List[Vec2D] = []  # 偏移折线全部顶点 [start, j1, j2, ..., end]
    prev_diff: Line2D | None = None
    prev_seg: Line2D | None = None
    printlog.info("构造差分折线:")
    for v in sVecList.GetList():
        diff = Line2D.make_parallel(v, vec_distance)
        printlog.info(f"  +平行线 {diff.start_pt}->{diff.end_pt}")

        if prev_diff is None or prev_seg is None:
            points.append(diff.start_pt)
        elif Vec2D.get_parallel(prev_seg.get_vec(), v.get_vec()) == 0:
            # 非共线段: 求两条平行线交点 修正上一顶点
            pt = Line2D.get_linear_junction(prev_diff, diff)
            assert pt is not None, "错误(两条虚拟差分线不存在线性交点)"
            points[-1] = pt
            printlog.info(f"  +交点 ({pt.x},{pt.y})")
        else:
            # 共线段: 平行线重合/平行 无交点 保持上一顶点(合并)
            printlog.info("  +共线段合并(无交点)")

        points.append(diff.end_pt)
        prev_diff = diff
        prev_seg = v

    # 导出有序点表 结构 [j1, j2, ..., 末段终点] 无起点(起点由头部参考差分线提供)
    ptList = VecList2D()
    for p in points[1:]:
        ptList.Append(p)
        printlog.info(f"  +点{ptList.Count()} ({p.x},{p.y})")

    return ptList


class BoardLike(Protocol):
    def Add(self, obj) -> None: ...
    def Remove(self, obj) -> None: ...


def InstanceNewDiff(
    ptList: VecList2D,
    diffStart: Tuple[TrackPoint2i, TrackPoint2i],
    diffEnd: Tuple[TrackPoint2i, TrackPoint2i] | None,
    info: ExportInfo_Result,
    board_like: BoardLike,
    make_track: Callable[[Vec2D, Vec2D], TrackLike],
):
    printlog.info("")
    printlog.info(f"{InstanceNewDiff.__name__}():")

    # 转换差分折线 开始于参考差分线(头)的起点
    pl = TrackPolyline2i(diffStart[0])
    printlog.info("构造差分折线:")
    printlog.info(f"  +固定起点{pl.GetPointCount()} {diffStart[0]}")
    pl.AddPoint(diffStart[1])
    printlog.info(f"  +调整端点{pl.GetPointCount()} {diffStart[1]}")

    # 端点列表的结构 无起点 仅有 所有交点 + 终点

    # 终点仅用于单差分对输入 作为最后那根新线路的终点
    # 必须弹出端点列表 不能参与建立新线路 以避免生成多一根新线路
    pts = ptList.GetList()
    ptEnd_indep = pts.pop()

    # 双差分对输入时 倒数第二点是已存在
    # 必须弹出端点列表 以避免生成多一根新线路
    ptEndPrve: Vec2D | None = pts.pop() if diffEnd is not None else None

    for i, pt in enumerate(pts):
        # 折线的终点 视为当前遍历交点的上一点
        prvePt = pl.GetEnd()
        assert prvePt.BindCount() == 1, f"错误(非法差分折线 终点绑定了{prvePt.BindCount()}个线路)"
        # 应用交点到折线终点
        prvePt.SetXY(pt.x, pt.y)

        # 本段新线路的终点 由下一个交点直接给定 末段在循环结束后独立设置
        # 不再使用 (1,0) 占位端点 避免残留 1 单位线段
        if i + 1 < len(pts):
            ptEnd = pts[i + 1]
        elif ptEndPrve is not None:
            ptEnd = ptEndPrve
        else:
            ptEnd = ptEnd_indep

        # 截断后与终点重合的退化线段 不参与建线
        if int(pt.x) == int(ptEnd.x) and int(pt.y) == int(ptEnd.y):
            printlog.info(f"  +跳过退化线段 ({int(pt.x)},{int(pt.y)})->({int(ptEnd.x)},{int(ptEnd.y)})")
            continue

        # 头部参考差分线终点被压到自身起点 将退化为零长线段 冗余线路直接移除
        if i == 0 and int(pt.x) == diffStart[0].x and int(pt.y) == diffStart[0].y:
            board_like.Remove(prvePt.GetBindFirst().obj)
            printlog.info("  +移除退化的头部参考差分线(交点与自身起点重合)")

        # 构建PCB线路 起点为交点 终点为下一交点(或独立设置的末段终点)
        pobj = make_track(pt, ptEnd)
        pobj.setWidth(info.width)
        pobj.SetLayer(info.layer)

        # 插入PCB线路
        pobj.AddTo(board_like)

        # 构建端点绑定信息
        StartBind = TrackPoint2i.bindInfo(pobj, TrackPoint2i.TRACK_START_POINT)
        EndBind = TrackPoint2i.bindInfo(pobj, TrackPoint2i.TRACK_END_POINT)

        # 把线路起点 绑定到折线目前的终点
        prvePt.AppendBind(StartBind)

        # 把线路终点 绑定到新的端点 后插入多边形末尾
        thisPt = TrackPoint2i(ptEnd.x, ptEnd.y, EndBind)
        pl.AddPoint(thisPt)
        printlog.info(f"  +新建端点{pl.GetPointCount()} {thisPt}")

        continue

    # 设置终点 折线的最后一点 应用到尾差分对的起点
    if diffEnd is not None:
        # 折线的终点 视为参考差分线(尾)的起点
        endPt = pl.GetEnd()
        assert endPt.BindCount() == 1, f"错误(非法差分折线 终点绑定了{endPt.BindCount()}个线路)"
        assert ptEndPrve is not None

        # 尾部参考差分线终点被压到自身另一端 将退化为零长线段 冗余线路直接移除
        if int(ptEndPrve.x) == diffEnd[1].x and int(ptEndPrve.y) == diffEnd[1].y:
            board_like.Remove(diffEnd[0].GetBindFirst().obj)
            printlog.info("  +移除退化的尾部参考差分线(交点与自身另一端重合)")

        # 参考差分线(尾)的起点 绑定到折线终点
        endPt.AppendBind(diffEnd[0].GetBindFirst())
        endPt.SetXY(ptEndPrve.x, ptEndPrve.y)
        # 参考差分线(尾)的终点 插入折线 以方便后续设置
        pl.AddPoint(diffEnd[1])
        printlog.info(f"  +固定终点{pl.GetPointCount()} {diffEnd[1]}")

    # 设置终点 交点列表的最后一点 应用到最后那根新线路
    else:
        endPt = pl.GetEnd()
        assert endPt.BindCount() == 1, f"错误(非法差分折线 终点绑定了{endPt.BindCount()}个线路)"
        # 在主循环中最后构建的PCB线路 依然保持在等待设置的状态
        # 交点列表的最后一点 视为终点 设置到折线终点的坐标
        endPt.SetXY(ptEnd_indep.x, ptEnd_indep.y)

    # 更新折线内所有端点到绑定的PCB线路
    printlog.info("差分线段列表:")
    for pt in pl.GetList():
        printlog.info(f" 更新线路 {pt}")
        pt.Update()

    return pl


class TrackSource(Protocol):
    def GetSelectedTracks(self) -> List[TrackLike]: ...


def run(
    source: TrackSource,
    board_like: BoardLike,
    make_track: Callable[[Vec2D, Vec2D], TrackLike],
) -> Tuple[TrackPolyline2i | None, TrackPolyline2i | None, ExportInfo_Result | None]:
    inputList: List[TrackLike] = source.GetSelectedTracks()

    if len(inputList) < 3:
        return None, None, None

    # 解析输入的线路 线路 > 端点 > 折线
    infoResult = ExportInfo(inputList)
    pointResult = ExportPoint(inputList)
    lineResult = ExportLine(pointResult)

    # 单端折线 转换到向量表
    vecList = PolylineToVecList(lineResult.sReferPolyline)
    assert vecList.Count() > 0, "错误(参考折线无有效线段)"

    # 校验 起始侧 参考差分线和单端折线 的差分关系正确性
    vecList.pMoveStart()
    polarStart, distanceStart = CheckPairPolar(
        vecList.GetCurrent(),
        lineResult.dReferStart,
    )
    if polarStart == 1:
        diffStart = lineResult.dReferStart
    else:
        diffStart = (lineResult.dReferStart[1], lineResult.dReferStart[0])

    # 校验 结束侧 参考差分线和单端折线 的差分关系正确性

    if lineResult.dReferEnd is not None:
        vecList.pMoveEnd()
        polarEnd, distanceEnd = CheckPairPolar(
            vecList.GetCurrent(),
            lineResult.dReferEnd,
        )
        assert abs(abs(distanceStart) - abs(distanceEnd)) < 10, f"失败(头尾的差分间距不一致) 间距差={distanceStart - distanceEnd}"

        if polarEnd == 1:
            diffEnd = lineResult.dReferEnd
        else:
            diffEnd = (lineResult.dReferEnd[1], lineResult.dReferEnd[0])
    else:
        diffEnd = None

    # 有符号差分线距 负号代表参考差分线在单端折线的逆角度方向
    distance = int(distanceStart)

    # 通过差分线距 生成新差分折线交点列表
    ptList = GenerateNewPointList(vecList, distance)

    # 更新交点 新建PCB线路
    refer_pl = lineResult.sReferPolyline
    diff_pl = InstanceNewDiff(
        ptList,
        diffStart,
        diffEnd,
        infoResult,
        board_like,
        make_track,
    )

    return refer_pl, diff_pl, infoResult
