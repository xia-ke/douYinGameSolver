"""Layer runners for active real replay fixtures.

Register a runner here when a pending case is promoted to active. Keeping runner
registration in one module makes active coverage explicit and prevents a manifest-only
status change from silently passing.
"""

# Intentionally empty until the first real visual fixture is promoted to active.
# Example:
#
# from .harness import register_runner
#
# @register_runner("parking_ocr")
# def run_parking_ocr(case, repo_root):
#     ...
