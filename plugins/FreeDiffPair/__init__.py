import os
from pathlib import Path
import sys
import traceback

PLUGIN_DIR = Path(__file__).parent.as_posix()
sys.path.append(os.path.dirname(PLUGIN_DIR))

try:
    from _logging import PrefixLogger, Config

    G_PLUGIN_LOG_FILE = f"{PLUGIN_DIR}/plugin.log"
    logger = PrefixLogger(
        "Plugin",
        sinks=[
            Config.ColorStdoutSink(),
            Config.ColorFileSink(G_PLUGIN_LOG_FILE, mode="w"),
        ],
        dyer=Config.default_dyer,
    )
    logger.info(f"load plugin {PLUGIN_DIR}")

except Exception as e:
    with open(f"{PLUGIN_DIR}/plugin.log", "a") as f:
        f.write(traceback.format_exc())
    raise e


try:
    from ._main import FreeAngleDifferentialPair

    FreeAngleDifferentialPair().register()

except Exception as e:
    for i in traceback.format_exc().split("\n"):
        logger.fatal(i)
    raise e
