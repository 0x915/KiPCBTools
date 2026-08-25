import os
import sys
import timeit
from typing import List

import pcbnew

from .VecSolver import run
from ._kicadlib import PY_PCB_TRACK, AddCuShapeTrackLock

from . import logger as printlog


def wxPrint(msg):
    import wx

    wx.LogMessage(msg)


class FreeAngleDifferentialPair(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "FreeDiffPair v08"
        self.category = "pcbnew"
        self.description = "Generate free angle differential pairs - 0x915"
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "./main.png")
        self.show_toolbar_button = True

    def Run(self):
        try:
            board = pcbnew.GetBoard()

            class _Source:
                def GetSelectedTracks(self) -> List[PY_PCB_TRACK]:
                    ret: List[PY_PCB_TRACK] = []
                    for track in board.GetTracks():
                        if not isinstance(track, pcbnew.PCB_TRACK):
                            raise TypeError
                        if track.IsSelected():
                            ret.append(PY_PCB_TRACK(track))
                    return ret

            def make_track(start, end):
                return PY_PCB_TRACK((board, start, end))

            printlog.debug(
                f"插件耗时 {timeit.timeit(lambda: self._execute(_Source(), board, make_track), number=1):.3f}s\n",
            )

        except AssertionError as e:
            printlog.fatal(f"{e}")
            wxPrint(f"{e}")
            return

        except Exception as e:
            raise e

        pass

    def _execute(self, source, board, make_track):
        refer_pl, diff_pl, infoResult = run(source, board, make_track)
        if infoResult is None:
            return
        assert refer_pl is not None and diff_pl is not None
        # AddCuShapeTrackLock(refer_pl, infoResult, board)
        # AddCuShapeTrackLock(diff_pl, infoResult, board)
        pass
