import math

from MathLib.Define import TOLCF2, WithinTolerance
from MathLib.Radian import Rad
from MathLib.Vector import Vec2D
import random


def _Vector2D_Test():
    from spdlogger import LoggerUtil

    logger = LoggerUtil.Logger("Test", [])

    def TestRotationError():
        print()
        logger.info("------------ TestRotationError ------------")
        vec = Vec2D(1, 0)
        deg1 = Rad.fromDeg(1)
        r = vec.Clone()
        for i in range(1, 361):
            r.Rotate(deg1)
            y1 = 1 - r.x * r.x
            d1 = math.sqrt(abs(abs(r.y**2) - abs(y1)))
            logger.debug(f"vAdd = {r}  {d1:.20f}")

            v = vec.Clone()
            v.Rotate(v=Rad.fromDeg(i))
            y2 = 1 - v.x * v.x
            d2 = math.sqrt(abs(abs(v.y**2) - abs(y2)))
            logger.debug(f"vSet = {v}  {d2:.20f}")

            d3 = abs(d2 - d1)
            rA = r.angle()
            vA = v.angle()

            logger.debug(f"rAdd {rA}")
            logger.debug(f"rSet {vA}")
            logger.track(f"     {d3:.24f}")

            if i == 360:
                i = 0
            if (
                abs(abs(rA.GetDeg()) - i) <= 0.000001  #
                and abs(abs(vA.GetDeg()) - i) <= 0.000001
            ):
                logger.info(f"     PASS {i} DEG")
                continue
            if abs(abs(rA.GetDeg()) - i) > 0.000001:
                logger.error(f"     FAILED ADD DEG {abs(abs(rA.GetDeg()) - i)}")
            if abs(abs(vA.GetDeg()) - i) > 0.000001:
                logger.error(f"     FAILED SET DEG {abs(abs(vA.GetDeg()) - i)}")

    def TestSinCos():
        print()
        logger.info("------------ TestSinCos ------------")
        raw = Vec2D(1, 0)
        for i in range(1, 361):
            vec = raw.Clone()
            vec.SetAngle(v=Rad.fromDeg(i))
            ra = vec.angle()
            logger.debug(f"Vec = {vec}")
            logger.debug(f"vRA = {ra}")

            sinv = vec.sin()
            cosv = vec.cos()
            sina = math.sin(ra.Get())
            cosa = math.cos(ra.Get())
            diff_sin = abs(sinv - sina)
            diff_cos = abs(cosv - cosa)

            logger.debug(f"Vec  sin {sinv:+.10f}  cos {cosv:+.10f}")
            logger.debug(f"vRA  sin {sina:+.10f}  cos {cosv:+.10f}")
            logger.track(f"         {diff_sin:+.10f}      {diff_cos:+.10f}")

            if diff_sin <= 0.000001 and diff_cos <= 0.000001:
                logger.info(f"          PASS {i} DEG")
                continue
            if diff_sin >= 0.000001:
                if WithinTolerance(sinv, sina, 0.000001):
                    logger.error(f"         FAILED SIN {i} 负号")
                else:
                    logger.error(f"         FAILED SIN {i} 错误的值")
            if diff_cos >= 0.000001:
                if WithinTolerance(cosv, cosa, 0.000001):
                    logger.error(f"         FAILED COS {i} 负号")
                else:
                    logger.error(f"         FAILED COS {i} 错误的值")

    def TestSetSize():
        print()
        logger.info("------------ Test SetSize ------------")
        vec = Vec2D(1, 1)
        for _ in range(10):
            i = random.randint(1, 99) + random.randint(1, 99) / 100
            vec.SetNorm(i)
            d = abs(i - vec.norm())
            logger.debug(f"SetNorm {vec}  {d:.20f}")
            if abs(i - abs(vec.norm())) <= 0.000001:
                logger.track(f"PASS SET NORM {i:.2f} UNIT")
                continue
            logger.error(f"FAILED SET NORM {i:.2f} UNIT")

    def TestVec2dLine():
        print()
        logger.info("------------ Test Vec2D Line ------------")
        a = Vec2D.fromXYXY(Vec2D(0, 0), Vec2D(1, 0))
        b = Vec2D.fromXYXY(Vec2D(1, 1), Vec2D(3, 3))
        c = Vec2D.fromXYXY(Vec2D(1, 1), Vec2D(2, 3))
        logger.info(f"a = {a}")
        logger.info(f"b = {b}")
        logger.info(f"c = {c}")
        logger.debug(f"Line.GetPointDistance( a, b.GetStart() ) = {Vec2D.GetPointDistance(a, b.GetStart()):.20f}")
        logger.debug(f"Line.GetPointDistance( a, b.GetEnd()   ) = {Vec2D.GetPointDistance(a, b.GetEnd()):.20f}")
        logger.debug(f"Line.GetProjection(a, b) = {Vec2D.GetProjectionSize(a, b):.20f}")
        logger.debug(f"Line.GetParallel(a, c) = {Vec2D.GetParallel(a, c).name}")
        d = Vec2D.MakeParallelLine(a, 1)
        logger.info(f"d = {d}")
        logger.info(f"a = {a.formatPoint()}")
        logger.info(f"d = {d.formatPoint()}")
        logger.debug(f"Line.GetPointDistance( a, d.GetStart() ) = {Vec2D.GetPointDistance(a, d.GetStart()):.20f}")
        logger.debug(f"Line.GetPointDistance( a, d.GetEnd()   ) = {Vec2D.GetPointDistance(a, d.GetEnd()):.20f}")

    def TestVec2DMakeParallel():
        print()
        logger.info("------------ Test Vec2D MakeParallel ------------")
        def TestMakeParallel(a, clearance):
            logger.debug(f"a  = {a}")
            pl = Vec2D.MakeParallelLine(a, clearance)
            logger.debug(f"pl = {pl}")
            dStart = Vec2D.GetPointDistance(a, pl.GetStart())
            dEnd = Vec2D.GetPointDistance(a, pl.GetEnd())
            logger.info(f"dStart = {dStart:.20f}  {dStart-clearance:+.20f}")
            logger.info(f"dEnd   = {dEnd:.20f}  {dEnd-clearance:+.20f}")
            if not TOLCF2(dStart,"==",dEnd,0.0001):
                logger.error("   FAILED 不平行")
                return
            if not TOLCF2(dStart,"==",clearance,0.0001):
                logger.error("   FAILED 距离错误")
                return
            match Vec2D.GetParallel(a,pl) :
                case Vec2D.RET.Parallel.Negative:
                    logger.error("   FAILED 方向相反")
                case Vec2D.RET.Parallel.Positive:
                    logger.track(f"   PASS  {a.angle().GetDeg():.6f} DEG  Clearance={clearance}")



        vec = Vec2D(10, 0)
        for i in range(1, 360):
            v = vec.Clone()
            v.SetAngle(Rad.fromDeg(i))
            TestMakeParallel(v,random.randint(1, 10))


    def TestVec2DCollinear():
        print()
        logger.info("------------ Test Vec2D Collinear ------------")
        def TestGetCollinear(a, b):
            logger.debug(f"a = {a}")
            logger.debug(f"b = {b}")
            ret = Vec2D.GetCollinear(a, b)
            logger.info(f"GetRelation(a,b) = {ret.name} / 0b{ret:b}")

        short = Vec2D(1, 1)
        long = Vec2D(4, 4)
        long.offset = Vec2D(2, 2)
        for i in range(16):
            short.SetStart(i / 2, i / 2)
            TestGetCollinear(short, long)

    # TestRotationError()
    # TestSinCos()
    # TestSetSize()
    # TestVec2dLine()
    # TestVec2DCollinear()
    TestVec2DMakeParallel()
    pass


if __name__ == "__main__":
    _Vector2D_Test()
