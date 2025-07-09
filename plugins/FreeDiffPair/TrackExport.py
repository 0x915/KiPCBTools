import math
from .include import G_PLUGIN_LOG_FILE
from .logger import getLogger
from .mathLib import Vec2D
from .kiLib import PY_PCB_TRACK

from typing import List, Tuple

logger = getLogger("track-export")
logger.addFileHandler(G_PLUGIN_LOG_FILE)


class TPoint2i:
    TRACK_END_POINT = 1
    TRACK_START_POINT = -1

    class bindInfo:
        def __init__(self, pyTrack: PY_PCB_TRACK, ptype: int):
            self.obj: PY_PCB_TRACK = pyTrack
            self.ptype: int = ptype

    def __init__(self, x: int | float, y: int | float, bind: bindInfo) -> None:
        # 端点的坐标
        self._x: int = int(x)
        self._y: int = int(y)
        # 端点绑定的线路列表
        self._bindList: List[TPoint2i.bindInfo] = [bind]

    # fmt:off

    def SetXY(self,x:int|float,y:int|float):
        self._x = int(x)
        self._y = int(y)

    @property
    def x(self): return self._x

    @x.setter
    def x(self,x:int|float): self._x = int(x)

    @property
    def y(self): return self._y

    @y.setter
    def y(self,y:int|float): self._y = int(y)

    # fmt:on

    def AppendBind(self, i: bindInfo):
        if i in self._bindList:
            return False
        self._bindList.append(i)
        return True

    def RemoveBind(self, i: bindInfo):
        if i not in self._bindList:
            return False
        self._bindList.remove(i)
        return True

    def HasBind(self, obj: PY_PCB_TRACK, ptyle: int):
        for i in self._bindList:
            if (i.obj is obj) and (i.ptype == ptyle):
                return True
            continue
        return False

    def BindCount(self):
        return len(self._bindList)

    def GetBindList(self):
        return self._bindList

    def GetBindFirst(self):
        return self._bindList[0]

    # 更新所有绑定的线路对象
    def Update(self):
        if len(self._bindList) == 0:
            logger.error(f"{self} 端点未绑定任何线段 不会执行任何操作")
            return

        for track in self._bindList:
            tobj = track.obj
            ptype = track.ptype
            # 端点类型终点
            if ptype == self.TRACK_END_POINT:
                logger.info(f"(0x{id(tobj):X}).SetEnd   ({self._x:+}, {self._y:+})")
                tobj.SetEnd(Vec2D(self._x, self._y))
                continue
            # 端点类型起点
            if ptype == self.TRACK_START_POINT:
                logger.info(f"(0x{id(tobj):X}).SetStart ({self._x:+}, {self._y:+})")
                tobj.SetStart(Vec2D(self._x, self._y))
                continue
            logger.error(f"<0x{id(tobj):X}> 不存在的端点类型({ptype}) 不会执行任何操作.")
        return

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} 0x{id(self):X} XY=({self._x:+},{self._y:+}) bind:{len(self._bindList)}>"

    def __repr__(self) -> str:
        return self.__str__()

    def XYEQ(self, x: int, y: int):
        return self.x == x and self.y == y


class Polyline2D:
    def __init__(self, start: TPoint2i) -> None:
        self._i: int = 1
        self._ptList: List[TPoint2i] = [start]

    def Reverse(self):
        self._ptList.reverse()

    def GetList(self):
        return self._ptList

    def GetPointCount(self):
        return len(self._ptList)

    def GetStart(self):
        return self._ptList[0]

    def GetEnd(self):
        return self._ptList[-1]

    def AddPoint(self, pt: TPoint2i):
        if pt in self._ptList:
            return False
        self._ptList.append(pt)
        return True

    def RemovePoint(self, pt):
        if pt not in self._ptList:
            return False
        self._ptList.remove(pt)
        return True

    def _GetLine(self, i: int):
        return self._ptList[i - 1], self._ptList[i]

    def GetCurrent(self):
        if self.GetPointCount() <= 1:
            return None
        return self._GetLine(self._i)

    def GetPrev(self):
        if self._i <= 1:
            return None
        return self._GetLine(self._i - 1)

    def GetNext(self):
        if self._i >= self.GetPointCount() - 1:
            return None
        return self._GetLine(self._i + 1)

    def pMoveNext(self):
        if self._i >= self.GetPointCount() - 1:
            return False
        self._i += 1
        return True

    def pMovePrev(self):
        if self._i <= 1:
            return False
        self._i -= 1
        return True

    def pMoveStart(self):
        self._i = 1

    def pMoveEnd(self):
        self._i = self.GetPointCount() - 1

    def pGet(self):
        return self._i

    def pSet(self, p: int):
        if self._i >= self.GetPointCount():
            raise OverflowError
        if self._i <= 1:
            raise ValueError
        self._i = p

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} 0x{id(self):X} point:{self.GetPointCount()}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self):
        index = 1
        while not index >= len(self._ptList):
            yield self._GetLine(index)
            index += 1


class ExportInfo_Result:
    def __init__(
        self,
        layer: int,
        width: int,
    ) -> None:
        self.layer: int = layer
        self.width: int = width


def ExportInfo(pyTrackList: List[PY_PCB_TRACK]):
    #
    logger.info("")
    logger.info(f"{ExportInfo.__name__}():")

    logger.info("获取线路物理信息:")
    layer = pyTrackList[0].GetLayer()
    width = pyTrackList[0].GetWidth()
    for track in pyTrackList[1:]:
        ret_layer = track.GetLayer()
        ret_width = track.GetWidth()
        if layer != ret_layer:
            assert "失败(线路不同层)"
        if width != ret_width:
            assert "失败(线路宽度不一致)"

    logger.info(f"  铜层 {pyTrackList[0].GetLayerName()}({layer})")
    logger.info(f"  线宽 {width}(unit)")
    return ExportInfo_Result(layer, width)


class ExportPoint_Result:
    def __init__(
        self,
        diff_tlist: List[Tuple[TPoint2i, TPoint2i]],
        share_ptlist: List[TPoint2i],
        share_ep_2pt: Tuple[TPoint2i, TPoint2i],
    ) -> None:
        self.diffList: List[Tuple[TPoint2i, TPoint2i]] = diff_tlist
        self.shareList: List[TPoint2i] = share_ptlist
        self.shareEp2x: Tuple[TPoint2i, TPoint2i] = share_ep_2pt


def ExportPoint(pyTrackList: List[PY_PCB_TRACK]):
    #
    all_point_list: List[TPoint2i] = []
    logger.info("")
    logger.info(f"{ExportPoint.__name__}():")

    def AddPoint(x, y, ptInfo: TPoint2i.bindInfo, ptList: List[TPoint2i]):
        for point in ptList:
            # 端点存在 绑定线路
            if point.XYEQ(x, y):
                point.AppendBind(ptInfo)
                logger.info(f"  绑定到   {point}")
                return
            continue
        # 增加到缓存 绑定线路
        obj = TPoint2i(x, y, ptInfo)
        ptList.append(obj)
        logger.info(f"  新的端点 {obj}")

    # 获取 所有线段端点 合并相同点时绑定到对应对象
    logger.info(f"输入 线路端点检查(x{len(pyTrackList)}):")
    for track in pyTrackList:
        logger.info(f"  {track}")
        # 导入线路的起点
        ptStart = track.GetStart()
        AddPoint(
            ptStart.x,
            ptStart.y,
            TPoint2i.bindInfo(track, TPoint2i.TRACK_START_POINT),
            all_point_list,
        )
        # 导入线路的终点
        ptEnd = track.GetEnd()
        AddPoint(
            ptEnd.x,
            ptEnd.y,
            TPoint2i.bindInfo(track, TPoint2i.TRACK_END_POINT),
            all_point_list,
        )

    logger.info("端点捕获列表:")
    indep_point_list: List[TPoint2i] = []
    share_point_list: List[TPoint2i] = []

    # 导出 折线端点列表 独立线列表
    for point in all_point_list:
        logger.info(f"  {point}")
        count = point.BindCount()
        assert count <= 2, "端点被超过两根线共享."
        if count == 1:
            indep_point_list.append(point)
        elif count == 2:
            share_point_list.append(point)
        continue

    # 差分参考线加上参考多段线 最多有六个悬空端点(多段线x2 + 头差分线x2 + 尾差分线x2)
    # 最少需要一根差分参考线 至少需要四个悬空端点(多段线x2 + 差分线x2)
    assert len(share_point_list) != 0, "失败(不存在连续的多段线)"
    assert len(indep_point_list) == 4 or len(indep_point_list) == 6, "失败(捕获不到四或六个悬空端点)"

    track_diff_list: List[Tuple[TPoint2i, TPoint2i]] = []

    def FindALine(ptList: List[TPoint2i]):
        ret: Tuple[TPoint2i, TPoint2i] | None = None
        # 在点列表中查找同一线段上的两端点
        for pt1 in ptList:
            for pt2 in ptList:
                # 跳过自己
                if pt1 is pt2:
                    continue
                # 端点不在同一线段 未命中参考差分线
                if pt1.GetBindFirst().obj is not pt2.GetBindFirst().obj:
                    continue
                # 端点属性不能相同 (设计上不该相同)
                if pt1.GetBindFirst().ptype == pt2.GetBindFirst().ptype:
                    continue
                # 仅记录找到的第一条线段 然后从输入列表中移除两端点
                ret = (pt1, pt2)
                ptList.remove(pt1)
                ptList.remove(pt2)
                # 立刻返回
                return ret
            continue
        # 找不到任何线段
        return None

    line_count = len(pyTrackList)

    # 导出 第一根 差分参考线
    track1d_2pt = FindALine(indep_point_list)
    assert track1d_2pt is not None, "失败(未找到参考差分线)"

    # 插入第一参考线列表
    line_count -= 1
    track_diff_list.append(track1d_2pt)
    logger.info("输出 参考差分线A:")
    logger.info(f"  {track1d_2pt[0].GetBindFirst().obj}")
    for point in track1d_2pt:
        logger.info(f"  {point}")

    # 导出 第二根 差分参考线
    track2d_2pt = FindALine(indep_point_list)
    if track2d_2pt is not None:
        line_count -= 1
        # 插入第二参考线列表
        track_diff_list.append(track2d_2pt)
        logger.info("输出 参考差分线B:")
        logger.info(f"  {track2d_2pt[0].GetBindFirst().obj}")
        for point in track2d_2pt:
            logger.info(f"  {point}")

    # 设计上 独立的线段 只允许有最多两根(头差分线+尾差分线)
    assert FindALine(indep_point_list) is None, "失败(存在多余独立线段)"

    # 验证 折线端点的数量(总输入数量-参考差分线数量)
    logger.info(f"输出 参考单端共享端点(x{len(share_point_list)}):")
    for point in share_point_list:
        logger.info(f"  {point}")

    # 设计上 有N条折线 就有N-1个端点
    assert line_count - 1 == len(share_point_list), "失败(共享端点数量错误)"

    # 构造 唯一的单端折线的两个端点
    assert len(indep_point_list) == 2, "失败(悬空端点数量过多)"
    share_endpoint_2pt = indep_point_list[0], indep_point_list[1]

    logger.info(f"输出 参考单端悬空端点(x{len(share_endpoint_2pt)}):")
    for point in share_endpoint_2pt:
        logger.info(f"  {point}")

    return ExportPoint_Result(
        track_diff_list,
        share_point_list,
        share_endpoint_2pt,
    )


class ExportLine_Result:
    def __init__(
        self,
        polyline: Polyline2D,
        dReferStart: Tuple[TPoint2i, TPoint2i],
        dReferEnd: Tuple[TPoint2i, TPoint2i] | None,
    ) -> None:
        self.sReferPolyline: Polyline2D = polyline
        self.dReferStart: Tuple[TPoint2i, TPoint2i] = dReferStart
        self.dReferEnd: Tuple[TPoint2i, TPoint2i] | None = dReferEnd


def ExportLine(obj: ExportPoint_Result):
    #
    track_diff_list = obj.diffList
    share_point_list = obj.shareList
    share_enpoint_2pt = obj.shareEp2x

    logger.info("")
    logger.info(f"{ExportLine.__name__}():")

    # 验证 参考差分线数量
    tdiff_count = len(track_diff_list)
    assert tdiff_count == 1 or tdiff_count == 2, "错误(参考差分线不存在或超出支持数量)"

    def getLineToPointDistanceSum(line: Tuple[TPoint2i, TPoint2i], pt: TPoint2i):
        d1 = math.sqrt(math.pow(line[0].x - pt.x, 2) + math.pow(line[0].y - pt.y, 2))
        d2 = math.sqrt(math.pow(line[1].x - pt.x, 2) + math.pow(line[1].y - pt.y, 2))
        return int(d1 + d2)

    # 定义 第一条参考差分线为起点 第二条存在则为终点
    tdiff_start: Tuple[TPoint2i, TPoint2i] = track_diff_list[0]
    tdiff_end: Tuple[TPoint2i, TPoint2i] | None = None
    if len(track_diff_list) == 2:
        tdiff_end = track_diff_list[1]

    ep1 = share_enpoint_2pt[0]
    ep2 = share_enpoint_2pt[1]

    # 获取 端点 到 选定参考差分线两端点 的距离和
    # 距离和最小的端点 定义为 多端线的起点
    ptStart = ep1
    ptEnd = ep2
    if getLineToPointDistanceSum(tdiff_start, ep1) > getLineToPointDistanceSum(tdiff_start, ep2):
        ptStart = ep2
        ptEnd = ep1

    # 构建 参考单端折线 起点
    pl = Polyline2D(ptStart)
    logger.info(f"输入 从点列表(x{len(share_point_list) + len(share_enpoint_2pt)})构建多边形:")
    logger.info(f"  +起点{pl.GetPointCount()} {ptStart}")

    # 折线的共享端点会绑定两个线对象
    # 找到对侧端点后 获取线对象时
    # 要过滤当前对象 以避免逆向寻找端点

    loop_break = len(share_point_list) + 2
    black_obj: PY_PCB_TRACK | None = None
    while True:
        # 循环保护
        loop_break -= 1
        assert loop_break > 0, "循环保护(构建折线时超出最大循环次数)"

        cur_bind: TPoint2i.bindInfo | None = None
        for bind in pl.GetEnd().GetBindList():
            # 设计上 最多会遍历出两个 绑定的线对象
            # 流程上 上一次使用的对象应该被屏蔽 以避免逆向搜索
            # 起点a0 对侧是终点a1 而a1对侧是b0或a0 不能搜索回a0 所以要屏蔽a0a1的线对象
            if bind.obj is black_obj:
                continue
            cur_bind = bind

        # 设计上 末端点只绑定了一个线对象 上一次使用后要求屏蔽 本次将无对象 循环应结束
        if cur_bind is None:
            break

        cur_pobj = cur_bind.obj
        cur_ptype = cur_bind.ptype
        cur_point: TPoint2i | None = None
        for pt in share_point_list:
            # 存在匹配对侧端点 ptNext会保持None
            if pt.HasBind(cur_pobj, -cur_ptype) is False:
                continue
            # 插入线段的另一侧端点
            pl.AddPoint(pt)
            cur_point = pt
            logger.info(f"  +端点{pl.GetPointCount()} {pt} → {cur_pobj}")
            break

        # 存在下个末端端点 重置循环
        if cur_point is not None:
            black_obj = cur_pobj
            continue

        # 设计上 总端点数 - 未插入的终点 = 线的数量
        assert pl.GetPointCount() - 1 == len(share_point_list), "输入线段存在连续性问题"

        break

    # 插入最后的端点
    pl.AddPoint(ptEnd)
    logger.info(f"  +终点{pl.GetPointCount()} {ptEnd} = {ptEnd.GetBindFirst().obj}")

    logger.info("输出 参考单端折线:")
    logger.info(f"  {pl}")

    logger.info("输出 起始参考差分线:")
    logger.info(f"  {tdiff_start[0]},{tdiff_start[1]}")

    if tdiff_end is not None:
        logger.info("输出 结束参考差分线:")
        logger.info(f"  {tdiff_end[0]},{tdiff_end[1]} ")

    return ExportLine_Result(
        pl,
        tdiff_start,
        tdiff_end,
    )
