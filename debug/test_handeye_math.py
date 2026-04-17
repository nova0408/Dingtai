from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    calibrate_hand_eye_ax_xb,
    evaluate_hand_eye_solution,
    generate_synthetic_motion_pairs,
)


def main() -> None:
    a_motions, b_motions, x_true = generate_synthetic_motion_pairs(
        sample_count=40,
        translation_scale=250.0,
        rotation_noise_deg=0.15,
        translation_noise=0.3,
        seed=7,
    )

    x_est = calibrate_hand_eye_ax_xb(a_motions, b_motions)
    residual = evaluate_hand_eye_solution(a_motions, b_motions, x_est)

    print("=== Ground Truth X ===")
    print(x_true.as_string(with_name=True))
    print("=== Estimated X ===")
    print(x_est.as_string(with_name=True))
    print("=== Residual ===")
    print(f"samples={residual.sample_count}")
    print(f"rotation_rmse_deg={residual.rotation_rmse_deg:.6f}")
    print(f"rotation_max_deg={residual.rotation_max_deg:.6f}")
    print(f"translation_rmse={residual.translation_rmse:.6f}")
    print(f"translation_max={residual.translation_max:.6f}")


if __name__ == "__main__":
    main()
