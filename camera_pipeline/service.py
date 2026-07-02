from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orin unified service entry")
    parser.add_argument("service", choices=("tray_detection", "opening_detection", "ball_pose_detection"))
    args, rest = parser.parse_known_args(argv)
    if args.service == "tray_detection":
        from .tray_detection.service import main as tray_detection_service_main

        return int(tray_detection_service_main(rest))
    if args.service == "ball_pose_detection":
        from .ball_pose_detection.service import main as ball_pose_detection_service_main

        return int(ball_pose_detection_service_main(rest))
    from .opening_detection.service import main as opening_detection_service_main

    return int(opening_detection_service_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
