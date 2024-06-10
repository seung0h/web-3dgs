from pathlib import Path
import argparse
from viewer import Viewer


def main(args) -> None:
    viewer = Viewer(Path(args.scene), args)
    viewer.update()

    while True:
        if viewer.update_check():
            viewer.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-scene", "-s", type=str, required=True, help="ply path of 3DGS"
    )
    parser.add_argument("-width", type=int, default=960)
    parser.add_argument("-height", type=int, default=540)

    args = parser.parse_args()
    main(args)
