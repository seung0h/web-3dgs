from pathlib import Path
import argparse
from viewer import Viewer


def main(args) -> None:
    viewer = Viewer(Path(args.scene), args)
    viewer.update()

    while True:
        if viewer.update_check():
            viewer.update()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scene', '-s', type=str, required=True, help='scene path(should be output of 3dgs)')

    args = parser.parse_args()
    main(args)
