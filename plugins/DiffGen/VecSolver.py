from typing import List, Tuple
import pcbnew

from .TrackExport import (
    GetRefPl,
    TrackPoint2i,
    ExportInfo,
    TrackInfo,
    ExportPoint,
    ExportPolyline,
)

from .include import G_PLUGIN_LOG_FILE


from .logger import getLogger


from .mathLib import (
    G_ANGLE_RAD_TOLERANCE,
    G_RAD_180_DEG,
    Rad,
    Line,
)

from .kiLib import PY_PCB_TRACK, XY2KiVECTOR2I, make_PCB_TRACK, make_SHAPE_CIRCLE  # noqa: F401
from .kiLib import toKiUnit, fromKiUnit  # noqa: F401

logger = getLogger("vec-solver")
logger.addFileHandler(G_PLUGIN_LOG_FILE)


G_RAD_P90DEG = Rad.fromDeg(90)
G_RAD_N90DEG = Rad.fromDeg(-90)


def PluginMain(board: pcbnew.BOARD):
    def GetInputTracks(board):
        ret: List[PY_PCB_TRACK] = []
        for track in board.GetTracks():
            if not isinstance(track, pcbnew.PCB_TRACK):
                raise TypeError
            if track.IsSelected():
                t = PY_PCB_TRACK(track)
                ret.append(t)
        return ret

    inputList: List[PY_PCB_TRACK] = GetInputTracks(board)

    if len(inputList) < 3:
        return

    # 解析PCB上选择的线路 线路 > 端点 > 折线
    trackInfo = ExportInfo(inputList)
    
    
    trackPointList = ExportPoint(inputList)
    trackPolylineList = ExportPolyline(trackPointList)
    refPl = GetRefPl(trackPolylineList)

    return
