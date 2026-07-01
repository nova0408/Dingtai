from __future__ import annotations

import argparse

from ..grasp_pose_pipeline.service import main as grasp_pose_pipeline_service_main
from ..tray_detection.service import main as tray_detection_service_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orin unified service entry")
    parser.add_argument("service", choices=("tray_detection", "grasp_pose_pipeline"))
    args, rest = parser.parse_known_args(argv)
    if args.service == "tray_detection":
        return int(tray_detection_service_main(rest))
    return int(grasp_pose_pipeline_service_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
