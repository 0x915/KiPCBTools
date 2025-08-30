import math
from typing import Iterator, List, Tuple

import kipy
from kipy import KiCad  # type: ignore
from kipy.board_types import BoardSegment, Track
from kipy.geometry import Vector2

from MathLib.Vector import Vec2D

# from TrackExport import KiSegmentPoint2D, TrackPolyline2D
# from kiLib import PY_PCB_TRACK
from spdlogger import logger


class KiLinePoint2D:
    """
    KiCad 线段端点(绑定BoardSegment对象)
    """

    class BindInfo:
        def __init__(self, line: BoardSegment, ptype: int):
            self.kiobj: BoardSegment = line
            self.btype: int = ptype

    BTYPE_START_POINT = -1
    BTYPE_END_POINT = 1

    def __init__(self, x: int | float, y: int | float) -> None:
        self.x: int | float = x
        self.y: int | float = y
        self._bindList: List[KiLinePoint2D.BindInfo] = []

    def AppendBind(self, i: BindInfo) -> bool:
        if i in self._bindList:
            return False
        self._bindList.append(i)
        return True

    def RemoveBind(self, i: BindInfo) -> bool:
        if i not in self._bindList:
            return False
        self._bindList.remove(i)
        return True

    def HasBind(self, obj: BoardSegment, btype: int):
        for bindinfo in self._bindList:
            if (bindinfo.kiobj is obj) and (bindinfo.btype == btype):
                return True
            continue
        return False

    def BindCount(self) -> int:
        return len(self._bindList)

    def GetBindList(self) -> List["KiLinePoint2D.BindInfo"]:
        return self._bindList

    def Update(self, board: kipy.kicad.Board):
        if len(self._bindList) == 0:
            logger.error(f"端点({self})未绑定线段.")
            return False
        for bindinfo in self._bindList:
            kiobj = bindinfo.kiobj
            btype = bindinfo.btype
            # 终点
            if btype == self.BTYPE_END_POINT:
                logger.info(f"更新线段({id(kiobj):X})终点坐标({self.x:+}, {self.y:+})")
                kiobj.end.from_xy(self.x, self.y)
                board.update_items(kiobj)
                continue
            # 起点
            if btype == self.BTYPE_START_POINT:
                logger.info(f"更新线段(0x{id(kiobj):X})起点坐标({self.x:+}, {self.y:+})")
                kiobj.start.from_xy(self.x, self.y)
                board.update_items(kiobj)
                continue
            logger.error(f"错误：线段(0x{id(kiobj):X})不存在的端点类型({btype}),不会执行任何操作.")
            continue
        return True

    def XYEQ(self, x: int | float, y: int | float):
        return self.x == x and self.y == y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        bind_str: str = ""
        for bind in self._bindList:
            bind_str += f" {id(bind.kiobj):X}:{'E' if bind.btype == self.BTYPE_END_POINT else 'S'}"
        return f"<{self.__class__.__name__} {id(self):X} ({self.x},{self.y}) {{bind{bind_str} {len(self._bindList)}}}>"


def ExportLinePoint(lineList: List[BoardSegment]) -> List[KiLinePoint2D]:
    logger.info("")
    logger.info(f"{ExportLinePoint.__name__}():")
    logger.info(f"提取(x{len(lineList)})输入线段列表中的端点:")

    def AddPoint(x: int, y: int, bindInfo: KiLinePoint2D.BindInfo, ptList: List[KiLinePoint2D]):
        # 端点存在
        for pt in ptList:
            if pt.XYEQ(x, y):
                pt.AppendBind(bindInfo)
                logger.info(f"  绑定   {pt}")
                return
            continue
        # 新的端点
        pt = KiLinePoint2D(x, y)
        pt.AppendBind(bindInfo)
        ptList.append(pt)
        logger.info(f"  新端点 {pt}")

    # 获取 所有线段的端点 并绑定到对应的线段
    allPointList: List[KiLinePoint2D] = []
    for line in lineList:
        logger.info(f"  {line}")
        # 线段起点
        AddPoint(
            line.start.x,
            line.start.y,
            KiLinePoint2D.BindInfo(line, KiLinePoint2D.BTYPE_START_POINT),
            allPointList,
        )
        # 线段终点
        AddPoint(
            line.end.x,
            line.end.y,
            KiLinePoint2D.BindInfo(line, KiLinePoint2D.BTYPE_END_POINT),
            allPointList,
        )

    logger.info("端点捕获列表:")

    # 导出 向量表端点列表 独立线列表
    for point in allPointList:
        logger.info(f"  {point}")
        if point.BindCount() > 2:
            logger.fatal(f"失败：端点({point})上有分支")
            for obj in point.GetBindList():
                logger.error(f"  {obj}")
            raise ValueError("端点被超过两根线共享,当前工作模式仅支持单根线或折线.")
        continue

    return allPointList


class KiLine2D:
    def __init__(self, start: KiLinePoint2D, end: KiLinePoint2D) -> None:
        self.start: KiLinePoint2D = start
        self.end: KiLinePoint2D = end

    def GetLength(self):
        return math.sqrt(  #
            math.pow(self.end.x - self.start.x, 2)  #
            + math.pow(self.end.y - self.start.y, 2)
        )

    def toVec2D(self):
        vec = Vec2D(0, 0)
        vec.SetEnd(self.end.x, self.end.y)
        vec.SetStart(self.start.x, self.start.y)
        return vec

    def __str__(self):
        return f"<{self.__class__.__name__} {id(self):X} Start({self.start.x},{self.start.y}) End({self.end.x},{self.end.y})>"


class KiLinePolyline2D:
    def __init__(self) -> None:
        self._p_line: int = 0
        self._pointlist: List[KiLinePoint2D] = []

    def CopyNew(self):
        new_pl = KiLinePolyline2D()
        for pt in self.PointIterator():
            new_pt = KiLinePoint2D(pt.x, pt.y)
            new_pl.InsertPoint(new_pt)
        return new_pl

    def MaxPointIndex(self) -> int | None:
        v = self.PointCount()
        # 端点指针只在有端点时存在
        return None if v == 0 else v - 1

    def MaxLineIndex(self) -> int | None:
        v = self.PointCount()
        # 线段指针只会在两个端点以上存在
        return None if v < 2 else v - 2

    def Reverse(self) -> None:
        return self._pointlist.reverse()

    def InsertPoint(self, pt: KiLinePoint2D, index: KiLinePoint2D | int | None = None) -> bool:
        # 重复插入
        if pt in self._pointlist:
            return False
        # type=KiLinePoint2D 插入到对象后
        if isinstance(index, KiLinePoint2D):
            if index not in self._pointlist:
                return False
            self._pointlist.insert(self._pointlist.index(index) + 1, pt)
            return True
        # type=int 插入到索引位置
        if isinstance(index, int):
            self._pointlist.insert(index, pt)
            return True
        # type=None 插入到列表末尾
        self._pointlist.append(pt)
        return True

    def RemovePoint(self, pt) -> bool:
        if pt not in self._pointlist:
            return False
        self._pointlist.remove(pt)
        return True

    def PointCount(self) -> int:
        return len(self._pointlist)

    def PointIterator(self):
        return self._pointlist.__iter__()

    def GetStartPoint(self) -> KiLinePoint2D | None:
        index_max = self.MaxPointIndex()
        if index_max is None:
            return None
        # 存在端点 才存在起点
        return self._pointlist[0]

    def GetEndPoint(self) -> KiLinePoint2D | None:
        index_max = self.MaxPointIndex()
        if index_max is None:
            return None
        # 当只有一个点时 起点同时也是终点
        return self._pointlist[index_max]

    def GetPoint(self, index: int) -> KiLinePoint2D | None:
        """返回 所选端点 异常时返回None"""
        v_max = self.MaxPointIndex()
        if v_max is None:
            return None
        if (index < 0) or (index > v_max):
            return None
        return self._pointlist[index]

    def GetLength(self):
        length = 0.0
        for line in self.LineIterator():
            length += line.GetLength()
        return length

    def LineCount(self):
        v = self.MaxLineIndex()
        return 0 if v is None else v + 1

    def LineIterator(self):
        index = 0
        index_max = self.MaxLineIndex()
        while True:
            if index_max is None:
                break
            if index > index_max:
                break
            line = self._Line(index)
            if line is None:
                break
            yield line
            index += 1

    def _Line(self, i: int) -> KiLine2D | None:
        """返回 端点(i) 和 端点(i+1) 构成的线段 异常将返回None"""
        start = self.GetPoint(i)
        end = self.GetPoint(i + 1)
        if start and end:
            return KiLine2D(start, end)
        return None

    def GetStartLine(self) -> KiLine2D | None:
        """返回 线段(p=0) 异常将返回None"""
        return self._Line(0)

    def GetPrevLine(self) -> KiLine2D | None:
        """返回 线段(p-1) 异常将返回None"""
        return self._Line(self._p_line - 1)

    def GetLine(self) -> KiLine2D | None:
        """返回 线段(p) 异常将返回None"""
        return self._Line(self._p_line)

    def GetNextLine(self):
        """返回 线段(p+1) 异常将返回None"""
        return self._Line(self._p_line + 1)

    def GetEndLine(self) -> KiLine2D | None:
        """返回 线段(pMax) 异常将返回None"""
        v = self.MaxLineIndex()
        if v is None:
            return None
        return self._Line(v)

    def SetLine(self, start: KiLinePoint2D | None = None, end: KiLinePoint2D | None = None) -> bool:
        """返回 当前线段 端点i 和 端点i+1 产生异常返回False"""
        if start is not None:
            s = self.GetPoint(self._p_line)
            if s is None:
                return False
            s.x = start.x
            s.y = start.y
        if end is not None:
            e = self.GetPoint(self._p_line + 1)
            if e is None:
                return False
            e.x = start.x
            e.y = start.y
        return True

    def pMoveStart(self):
        self._p_line = 0

    def pMovePrev(self) -> bool:
        p_min = 0
        if self._p_line - 1 < p_min:
            return False
        # 确保[索引在安全范围内]
        self._p_line -= 1
        return True

    def pGet(self):
        return self._p_line

    def pMoveNext(self):
        p_max = self.MaxLineIndex()
        if p_max is None:
            return False
        if self._p_line + 1 > p_max:
            return False
        # 确保[存在索引最大值]+[索引在安全范围内]
        self._p_line += 1
        return True

    def pMoveEnd(self) -> bool:
        p_max = self.MaxLineIndex()
        if p_max is None:
            return False
        # 确保[索引存在索引最大值]
        self._p_line = p_max
        return True

    def pSet(self, p: int):
        p_min = 0
        p_max = self.MaxLineIndex()
        if p < p_min or p > p_max:
            return False
        # 确保[索引在安全范围内]
        self._p_line = p
        return True

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} 0x{id(self):X} line={self.MaxLineIndex()} point={self.MaxPointIndex()}>"


def ExportLinePolyline(inPtList: List[KiLinePoint2D]):
    #
    logger.info("")
    logger.info(f"{ExportLinePolyline.__name__}():")
    logger.info(f"提取(x{len(inPtList)})端点列表中的折线:")

    def FindNextPoint(point: KiLinePoint2D, ptList: List[KiLinePoint2D]):
        # 搜索端点的每个绑定线路
        for cur_point in ptList:
            # 跳过自己
            if cur_point is point:
                continue
            # 查找另一侧端点
            for bind in point.GetBindList():
                if cur_point.HasBind(bind.kiobj, -bind.btype):
                    return cur_point
            continue
        return None

    def ExportPolyline2D(ptStart: KiLinePoint2D, ptList: List[KiLinePoint2D]):
        polyline = KiLinePolyline2D()

        polyline.InsertPoint(ptStart)
        logger.info(f"  +起点 {ptStart}")

        while True:
            # 查找 折线尾部端点 的 下一个关联端点
            ptNext = FindNextPoint(polyline.GetEndPoint(), ptList)
            # 结束折线
            if ptNext is None:
                break
            # 端点插入折线尾部
            polyline.InsertPoint(ptNext)
            # 移除已处理的端点 避免反向查找
            ptList.remove(ptNext)
            logger.info(f"  +端点 {ptNext}")

        if polyline.MaxLineIndex() is None:
            raise ValueError(f"错误：折线中没有线段，只有{polyline.MaxPointIndex()}个端点")

        return polyline

    plList: List[KiLinePolyline2D] = []

    loop_break = len(inPtList) + 10
    while True:
        # 循环保护
        loop_break -= 1
        assert loop_break > 0, "循环保护(构建向量表时超出最大循环次数)"

        # 解析完成
        if len(inPtList) == 0:
            break

        # 构建折线
        for pt in inPtList:
            # 只从独立点开始
            if pt.BindCount() != 1:
                continue
            inPtList.remove(pt)
            pl = ExportPolyline2D(pt, inPtList)
            plList.append(pl)
            logger.info(f"  = 折线{len(plList)} {pl}")
            break

    return plList


def MakeDiffPl(plin: KiLinePolyline2D, clearance: int | float):
    """创建 PlIn参考 Clearance间距 的平行线"""
    logger.info("")
    logger.info(f"{MakeDiffPl.__name__}():")
    logger.info(f"从折线{plin}创建<{clearance}>间距平行线:")

    plin.pSet(0)
    retpl = KiLinePolyline2D()

    while True:
        plin_cur_line = plin.GetLine()

        # 创建[当前参考线段]的平行线
        plin_cur_vec = plin_cur_line.toVec2D()
        plin_curdiff_vec = Vec2D.MakeParallelLine(plin_cur_vec, clearance)

        # 初始化起点
        if retpl.MaxLineIndex() is None:
            vec_start = plin_curdiff_vec.GetStart()
            pl_point = KiLinePoint2D(vec_start.x, vec_start.y)
            retpl.InsertPoint(pl_point)
            logger.info(f"  +起点 {pl_point}")

        # 修正[输出折线的终点]为[输出折线的末尾线段]与[当前参考线段平行线]的交点
        else:
            retpl_endline = retpl.GetEndLine()
            retpl_endline_vec = retpl_endline.toVec2D()
            # 构造新的平行线 求与上一线段交点

            junction = Vec2D.GetLinearJunction(plin_curdiff_vec, retpl_endline_vec)
            if junction is None:
                raise ValueError("失败：找不到交点")
            # 修正上一线段的终点到交点
            retpl_endline.end.x = junction.x
            retpl_endline.end.y = junction.y
            logger.info(f"   修正 {retpl_endline.end}")

        # 增加[当前参考线段平行线的终点]到[输出折线]
        vec_end = plin_curdiff_vec.GetEnd()
        ret_point = KiLinePoint2D(vec_end.x, vec_end.y)
        retpl.InsertPoint(ret_point)
        logger.info(f"  +端点 {ret_point}")

        # 移动[输入参考折线]线段索引到下一个
        if plin.pMoveNext():
            continue

        logger.info(f"  = 折线 {retpl}")

        break

    return retpl


#
#
#
#


def main(client: KiCad):
    board = client.get_board()

    ShapeLineList: List[BoardSegment] = []
    TrackList: List[Track] = []
    for i in board.get_selection():
        if isinstance(i, BoardSegment):
            ShapeLineList.append(i)
            logger.debug(f"+ {i}")
        elif isinstance(i, Track):
            TrackList.append(i)
            logger.debug(f"+ {i}")
        else:
            logger.warn(f"Skip {i}")


    if len(TrackList) == 0:
        logger.warn("未选择线路")
        return

    if len(TrackList) != 2:
        logger.warn("只能选择两根一对差分线")
        return

    if len(ShapeLineList) == 0:
        logger.warn("未选择图形线段")

    track1 = TrackList[0]
    track2 = TrackList[1]
    vec1 = Vec2D(0, 0)
    vec1.SetEnd(track1.end.x, track1.end.y)
    vec1.SetStart(track1.start.x, track1.start.y)
    vec2 = Vec2D(0, 0)
    vec2.SetEnd(track2.end.x, track2.end.y)
    vec2.SetStart(track2.start.x, track2.start.y)

    dp_layer = track1.layer
    dp_clearance = Vec2D.GetParallelClearance(vec1, vec2,10000000)

    # 检查差分线1 要求是平行线
    if dp_clearance is None:
        logger.fatal("输入线路不平行,要求输入平行差分对")
        logger.fatal(f"  {vec1} rA={vec1.angle()}")
        logger.fatal(f"  {vec2} rA={vec2.angle()}")
        logger.fatal(f"  平行误差 {Vec2D.Cross(vec1,vec2)/1000000}")
        return

    # 检查差分线2 要求有平行间距
    if dp_clearance == 0:
        logger.fatal("输入线路间距为0,要求输入正常差分对")
        logger.fatal(f"  {vec1}")
        logger.fatal(f"  {vec2}")
        return

    # 检查差分线3 要求相同线宽
    dp_width = track1.width
    if dp_width != track2.width:
        logger.fatal("输入线路宽度不相等,要求输入标准差分对")
        logger.fatal(f"  {track1}")
        logger.fatal(f"  {track2}")
        return

    logger.info("")
    logger.info(f"输入参考差分对:")
    logger.info(f"  {vec1}")
    logger.info(f"  {vec2}")
    logger.info(f"  dp_layer  = {dp_layer} ")
    logger.info(f"  dp_width  = {dp_width} ")
    logger.info(f"  clearance = {dp_clearance} ")

    # 导出所有连接点 并转化为(多条)折线
    ptList = ExportLinePoint(ShapeLineList)
    plList = ExportLinePolyline(ptList)

    # 生成的差分对与线段的距离只有输入间距的一半
    dp_center = int(dp_clearance / 2)

    kiobjList = []

    # 根据线段折线的轨迹生成新差分对
    for ref_pl in plList:
        logger.info("")
        logger.info(f"构造新差分对({ref_pl}):")

        pl_p = MakeDiffPl(ref_pl, dp_center)
        pl_n = MakeDiffPl(ref_pl, -dp_center)

        logger.info("")
        logger.info(f"增加任务({ref_pl}):")

        def makeTrack(plin: KiLinePolyline2D):
            for line in plin.LineIterator():
                new_track = Track()
                new_track.start = Vector2.from_xy(int(line.start.x), int(line.start.y))
                new_track.end = Vector2.from_xy(int(line.end.x), int(line.end.y))
                new_track.width = dp_width
                new_track.layer = dp_layer
                kiobjList.append(new_track)
                logger.info(f"  + {new_track}")
                continue
            pass

        makeTrack(pl_p)
        makeTrack(pl_n)
        continue

    # 提交请求到客户端
    commit = board.begin_commit()
    logger.info("")
    logger.info(f"提交到客户端({commit}):")
    for kiobj in kiobjList:
        board.create_items(kiobj)
        logger.info(f"  + {kiobj}")
    board.push_commit(commit,"MakeDiffTrack")


if __name__=='__main__':

    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")
    main(kicad)
