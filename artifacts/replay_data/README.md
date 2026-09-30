# Replay Data Artifacts

This directory stores pre-compiled historical replay datasets in Parquet format, corresponding to real operational windows extracted from Line 3 of the Bosch Production Line Performance dataset.

## Scenarios
- **Scenario A** (`scenario_A.parquet`): S29 sensor drift with co-occurring defect surge ($t \in [362.0, 386.0]$).
- **Scenario B** (`scenario_B.parquet`): ~50-hour production stoppage and restart defect burst ($t \in [492.0, 502.0]$).
- **Scenario C** (`scenario_C.parquet`): High-density line-wide failure surge in Week 44 ($t \in [730.0, 745.0]$).
- **Scenario D** (`scenario_D.parquet`): Week 51 plant shutdown and Week 52 cold restart ($t \in [850.0, 890.0]$).

## Regeneration
To regenerate all scenario datasets from raw exploration outputs:
```powershell
python -m twin.replay.build_replay_data --scenario ALL
```
Or for an individual scenario:
```powershell
python -m twin.replay.build_replay_data --scenario A
```
