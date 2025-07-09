import os
import wx
import sys
import pcbnew
import timeit
from .include import G_PLUGIN_LOG_FILE
from .VecSolver import PluginMain
from .logger import getLogger



logger = getLogger("main")
logger.addFileHandler(G_PLUGIN_LOG_FILE,'w')
logger.debug(f"写入文件 {os.path.realpath(G_PLUGIN_LOG_FILE)} ")
logger.debug("...\n\n")


def wxPrint(msg):
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
            logger.debug(
                f"插件耗时 {timeit.timeit(lambda: PluginMain(board), number=1):.3f}s\n",
            )

        except AssertionError as e:
            logger.fatal(f"{e}")
            wxPrint(f"{e}")
            return

        except Exception as e:
            raise e

        pass
