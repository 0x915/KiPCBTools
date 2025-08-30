import typing

Num = int | float
NumOrNone = Num | None


def StrNum(v: Num, floatdigit: int) -> str:
    if isinstance(v, int):
        return f"{v:+}"
    if isinstance(v, float):
        return f"{v:+.{floatdigit}f}"
    raise AssertionError(f"输入的 {type(v)} 类型 {v} 不是数字")



def TOLCF3(
    v1: Num,
    symA: typing.Literal["<", "<=", "==", ">=", ">", "!="],
    v2: Num,
    symB: typing.Literal["<", "<=", "==", ">=", ">", "!="],
    v3: Num,
    d: Num,
):
    return TOLCF2(v1, symA, v2, d) and TOLCF2(v2, symB, v3, d)


def TOLCF2(
    v1: Num,
    sym: typing.Literal["<", "<=", "==", ">=", ">", "!="],
    v2: Num,
    d: Num,
):
    match sym:
        case "<":
            return LT(v1, v2, d)
        case "<=":
            return LT(v1, v2, d)
        case "==":
            return EQ(v1, v2, d)
        case ">=":
            return GE(v1, v2, d)
        case ">":
            return GT(v1, v2, d)
        case "!=":
            return NE(v1, v2, d)
    raise AssertionError(f"符号 {sym} 不在[<,<=,==,>=,>,!=]内")


def LT(v1: Num, v2: Num, d: Num) -> bool:
    """
    False 当v1-v2在±=误差d内视为v1==v2  \n
    =True 误差之外v1-v2<0满足判定  \n
    False 误差之外v1-v2>0  \n
    """
    v1v2 = v1 - v2
    if abs(v1v2) <= d:
        return False
    return v1v2 < 0


def LE(v1: Num, v2: Num, d: Num) -> bool:
    """
    =True 当v1-v2在±=误差d内视为v1==v2满足判定  \n
    =True 误差之外v1-v2<0满足判定  \n
    False 误差之外v1-v2>0  \n
    """
    v1v2 = v1 - v2
    if abs(v1v2) <= d:
        return True
    return v1v2 < 0


def GT(v1: Num, v2: Num, d: Num) -> bool:
    """
    False 当v2-v1在±=误差d内视为v2==v1  \n
    =True 误差之外v2-v1<0满足判定  \n
    False 误差之外v2-v1>0  \n
    """
    return LT(v2, v1, d)


def GE(v1: Num, v2: Num, d: Num) -> bool:
    """
    =True 当v2-v1在±=误差d内视为v2==v1满足判定  \n
    =True 误差之外v2-v1<0满足判定  \n
    False 误差之外v2-v1>0  \n
    """
    return LE(v2, v1, d)


def EQ(v1: Num, v2: Num, d: Num) -> bool:
    """
    =True 当v2-v1在±=误差d内视为v2==v1满足判定  \n
    False 误差之外asb(v2-v1)>d  \n
    """
    if abs(v1 - v2) <= d:
        return True
    return False


def NE(v1: Num, v2: Num, d: Num) -> bool:
    """
    False 当v2-v1在±=误差d内视为v2==v1  \n
    =True 误差之外asb(v2-v1)>d满足判定  \n
    """
    return not EQ(v1, v2, d)


def GetRawTolerance(v1: Num, v2: Num) -> Num:
    return v1 - v2


def GetTolerance(v1: Num, v2: Num) -> Num:
    return abs(v1-v2)

def WithinTolerance(v1: Num, v2: Num, tolerance_le: Num) -> bool:
    return GetTolerance(v1, v2) <= tolerance_le
