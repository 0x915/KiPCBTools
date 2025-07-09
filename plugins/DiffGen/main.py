from kipy import KiCad  # type: ignore

if __name__=='__main__':

    kicad = KiCad()
    print(f"Connected to KiCad {kicad.get_version()}")

    board = kicad.get_board()
