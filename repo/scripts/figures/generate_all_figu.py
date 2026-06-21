"""Master script to generate all figures for the paper."""

import sys
import os
from pathlib import Path

# Add repo root to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import importlib
import traceback

FIGURE_SCRIPTS = [
    'figure_01_training_diagnostics',
    'figure_02_projection_efficacy',
    'figure_03_resolution_behavior',
    'figure_04_warmstart_performance',
    'figure_05_adaptive_fallback',
    'figure_06_pcg_cylinder',
    'figure_07_pcg_large_scale',
    'figure_08_drag_sensitivity',
    'figure_B09_rbf_profiles',
    'figure_C10_operator_gap',
    'figure_E11_field_comparison_cavity',
    'figure_E12_field_comparison_cylinder',
]


def main():
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating all figures for RBF-FD GNN Projection paper")
    print("=" * 60)

    success = []
    failed = []

    for script_name in FIGURE_SCRIPTS:
        print(f"\n{'-' * 40}")
        print(f"Generating: {script_name}")
        print(f"{'-' * 40}")

        try:
            module = importlib.import_module(f'scripts.figures.{script_name}')
            module.main()
            success.append(script_name)
            print(f"✓ SUCCESS: {script_name}")
        except Exception as e:
            failed.append((script_name, str(e)))
            print(f"✗ FAILED: {script_name}")
            print(f"  Error: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(success)}/{len(FIGURE_SCRIPTS)}")
    for s in success:
        print(f"  ✓ {s}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(FIGURE_SCRIPTS)}")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")

    print(f"\nOutput directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
