# Replay Regression Workflow

`tests/replay/` is the repository-local offline regression boundary for real game failures.
It intentionally separates two states:

- `active`: real source artifacts are preserved and executable assertions run.
- `pending_fixture`: expected historical behavior is documented, but the real source artifact is missing. Pending cases are reported in the pytest terminal summary and are never treated as passed visual evidence.

## Run

```bash
python -m pip install -r requirements-test.txt
pytest tests/replay/
```

The replay suite must not require an Android device or invoke ADB.

## Fixture contract

The manifest is `tests/replay/cases.json`. A case records:

- stable case id, subsystem, status and fixture kind;
- repository-relative real artifact paths;
- relevant action/vehicle inputs;
- structured expected OCR/board/parking/safety outputs;
- provenance and notes;
- a `pending_reason` while evidence is unavailable.

For a transition case, `artifacts` can reference the previous trusted stable screenshot/state, the next stable screenshot, and the existing `decision_log.txt`, `color_log.txt`, and `number_log.txt` files when relevant.

An `active` visual/transition case must point to real files that exist in the repository. It must also name an executable layer runner. `test_active_cases.py` fails if an active case has no registered runner, preventing a newly activated fixture from masquerading as coverage that only checks file existence.

## Capture a new real failure before fixing production logic

When a new real-device visual/rule failure appears:

1. Stop changing board thresholds, OCR rules, or strategy weights for that failure until its evidence is preserved.
2. Copy the real stable screenshot(s) involved into `tests/replay/fixtures/<case-id>/`. Prefer the solver's already-written stable screenshots; do not generate replacement imagery.
3. Copy the relevant diagnostic logs (`decision_log.txt`, `color_log.txt`, `number_log.txt`) when they are needed to establish the expected transition.
4. Record the executed action and any relevant vehicle/parking state in the manifest `inputs`.
5. Record only verified structured expectations in `expected` (for example OCR digits, removed cell coordinates/counts, parking remains, or stable-safety result).
6. Change the manifest case from `pending_fixture` to `active`, fill the real artifact paths, and provide/register the subsystem runner that returns structured actual results.
7. Run `pytest tests/replay/` offline. Temporarily alter one expected value and confirm the replay fails with a useful path, then revert that temporary edit.
8. Only after the failing replay exists should the affected production rule be changed. Keep the real artifacts with the regression permanently.

## Historical policy

The audit-era C01, C08, OCR-26, nearest-cell, and same-color low-remain screenshots were not preserved. Those historical pixel assertions remain `pending_fixture`; do not reconstruct or synthesize them.

The historical three-column game-screen requirement has now been promoted using a later naturally occurring real Level 16 three-column stable frame already preserved in `tests/replay/fixtures/level16_c01_64_before.png`, which is permitted by this workflow. A separate real Level 16 parking `16` frame is also preserved by `test_level16_parking_number_replay.py`.

The confirmed same-color low-remain domain rule also has a separate deterministic rule-level test. That test protects the rule implementation but is not a substitute for the missing historical screenshot replay.
