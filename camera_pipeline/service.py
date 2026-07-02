from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Orin unified service entry")
    parser.add_argument("service", choices=("camera_pipeline_service",))
    args, rest = parser.parse_known_args(argv)
    from .unified_service import main as camera_pipeline_service_main

    return int(camera_pipeline_service_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
