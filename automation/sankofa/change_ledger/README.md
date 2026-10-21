# Klapmuts change ledger, Dec 2025 -> Apr 2026 (21 Oct 2026)

Run order (system python3, PYTHONPATH=~/code/aru_sil_core/src for offline_vtr):
1. bag_change_ledger.py      — matched / moved / April-only / December-only on the ledger v5 registration
2. bag_change_rows.py        — bag rows from the matched bags; each change placed on a row, at a row end, or off the rows
3. bag_heights_lidar.py      — per-bag plant-top height from the lidar (aru_py_logger.LaserLogger, KISS-ICP/LIO poses, identity axes)
4. bag_growth.py             — height change over matched pairs; writes growth into change_ledger_dec_apr.csv
5. change_ledger_export.py   — the CSV with placement verdicts
Evidence: bag_provenance.py <April ids> (SAM3 grow-bag masks in the frames that see the spot), growth_evidence.py,
bag_change_traj.py (map with both keyframe paths), lidar_topdown_bags.py (frame check).
bag_change_stpg.py: the STPG-edge registration variant (windowed trimmed fits; 0.31 m bag agreement vs 0.24 m global — kept for the record).
Outputs under dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/.
