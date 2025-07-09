from typing import Tuple  # noqa: F401
import pcbnew

from .mathLib import Vec2D

kiUnit = 1000000


def toKiUnit(i: int | float) -> int:
    return int(i * kiUnit)


def fromKiUnit(i: int) -> float:
    return float(i / kiUnit)


class Value1iKI(int):
    def toKicadValue(self) -> int:
        return int(self)


class Value1fMM(float):
    def toKicadValue(self) -> int:
        return toKiUnit(float(self))


def XY2KiVECTOR2I(object) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(int(object.x), int(object.y))


def X2KiINT(object) -> int:
    return int(object)


class PY_PCB_TRACK:
    def __init__(self, obj: pcbnew.PCB_TRACK | Tuple[pcbnew.BOARD, Vec2D, Vec2D]) -> None:
        if isinstance(obj, pcbnew.PCB_TRACK):
            self.ki_pcb_track = obj
        elif isinstance(obj, tuple):
            board = obj[0]
            xy1 = obj[1]
            xy2 = obj[2]
            self.ki_pcb_track = make_PCB_TRACK(board, pcbnew.F_Cu, xy1, xy2)

        pass

    def AddTo(self, board: pcbnew.BOARD):
        board.Add(self.ki_pcb_track)

    def GetStart(self) -> Vec2D:
        ret: pcbnew.VECTOR2I = self.ki_pcb_track.GetStart()
        return Vec2D(ret.x, ret.y)

    def GetEnd(self) -> Vec2D:
        ret: pcbnew.VECTOR2I = self.ki_pcb_track.GetEnd()
        return Vec2D(ret.x, ret.y)

    def SetStart(self, v: Vec2D) -> None:
        self.ki_pcb_track.SetStart(XY2KiVECTOR2I(v))

    def SetEnd(self, v: Vec2D) -> None:
        self.ki_pcb_track.SetEnd(XY2KiVECTOR2I(v))

    def SetStartEnd(self, s: Vec2D, e: Vec2D) -> None:
        self.ki_pcb_track.SetStartEnd(XY2KiVECTOR2I(s), XY2KiVECTOR2I(s))

    def GetWidth(self) -> int:
        return self.ki_pcb_track.GetWidth()

    def setWidth(self, v):
        self.ki_pcb_track.SetWidth(X2KiINT(v))

    def GetLength(self) -> int:
        return self.ki_pcb_track.GetLength()

    def ApproxCollinear(self, t: "PY_PCB_TRACK") -> bool:
        return self.ki_pcb_track.ApproxCollinear(t.ki_pcb_track)

    def GetLayer(self) -> int:
        return self.ki_pcb_track.GetLayer()

    def GetLayerName(self):
        return self.ki_pcb_track.GetLayerName()

    def SetLayer(self, layer: int) -> int:
        return self.ki_pcb_track.SetLayer(layer)

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} 0x{id(self):X} "
            #
            + f"({self.GetStart().x:+},{self.GetStart().y:+})"
            + f"({self.GetEnd().x:+},{self.GetEnd().y:+}) "
            + f"W={self.GetWidth()}"
            + ">"
        )


def make_PCB_DIM_CENTER(
    parent: pcbnew.BOARD,
    layer,
    xy: Vec2D,
    size: int = toKiUnit(0.2),
    thickness_mm: int = toKiUnit(0.000001),
    angles: float | int = 0.0,
):
    kobj = pcbnew.PCB_DIM_CENTER(parent)
    kobj.SetLayer(layer)
    kobj.SetMirrored(pcbnew.IsBackLayer(layer))
    kobj.SetLineThickness(thickness_mm)
    kobj.SetStart(aPoint=XY2KiVECTOR2I(xy))
    kobj.SetEnd(aPoint=XY2KiVECTOR2I(xy + Vec2D(int(size / 2), 0)))
    kobj.Rotate(XY2KiVECTOR2I(xy), pcbnew.EDA_ANGLE(float(angles)))
    return kobj


def make_SHAPE_CIRCLE(
    parent: pcbnew.BOARD,
    layer,
    xy: Vec2D,
    size: int = toKiUnit(0.2),
    thickness: int = toKiUnit(0.0),
):
    kobj = pcbnew.PCB_SHAPE(parent)
    kobj.SetLayer(layer)
    kobj.SetFilled(True)
    kobj.SetShape(pcbnew.SHAPE_T_CIRCLE)
    kobj.SetWidth(thickness)
    kobj.SetCenter(XY2KiVECTOR2I(xy))
    kobj.SetRadius(int(size / 2))
    return kobj


def make_SHAPE_LINE(
    parent: pcbnew.BOARD,
    layer,
    xy1: Vec2D,
    xy2: Vec2D,
    thickness: int = toKiUnit(0.000001),
    angles: float | int = 0.0,
):
    kobj = pcbnew.PCB_SHAPE(parent)
    kobj.SetLayer(layer)
    kobj.SetWidth(thickness)
    kobj.SetStart(XY2KiVECTOR2I(xy1))
    kobj.SetEnd(XY2KiVECTOR2I(xy2))
    kobj.Rotate(XY2KiVECTOR2I(xy1), pcbnew.EDA_ANGLE(float(angles)))
    return kobj


def make_PCB_TRACK(
    parent: pcbnew.BOARD,
    layer,
    xy1: Vec2D,
    xy2: Vec2D,
    netcode=0,
    thickness: int = toKiUnit(0.000001),
    angles: float | int = 0.0,
):
    kobj = pcbnew.PCB_TRACK(parent)
    kobj.SetNetCode(netcode)
    kobj.SetLayer(aLayer=layer)
    kobj.SetWidth(thickness)
    kobj.SetStart(XY2KiVECTOR2I(xy1))
    kobj.SetEnd(XY2KiVECTOR2I(xy2))
    kobj.Rotate(XY2KiVECTOR2I(xy1), pcbnew.EDA_ANGLE(float(angles)))
    return kobj
