# Telemetry Dashboard

This starter gives you:
- **Sample telemetry** (`data/sample_indycar_telemetry.csv`): 2 laps at 20 Hz with realistic channels
- A minimal **MoTeC‑lite dashboard** script (`src/dashboard.py`) that:
  - Loads CSV
  - Plots speed and throttle/brake vs distance
  - Computes a basic lap delta (Lap B vs Lap A)

## Quickstart

```bash
# from the extracted folder
pip install -r requirements.txt
python src/dashboard.py --data data/sample_indycar_telemetry.csv --lap 1 --ref 1 --cmp 2
```

This will open three plots (one at a time): Speed vs Distance, Throttle/Brake vs Distance, and a Lap Delta overlay.

## CSV Columns
- `time_s`, `lap`, `distance_m`, `sector`
- `speed_mps`, `speed_kmh`
- `throttle_pct`, `brake_pct`
- `steer_deg`
- `engine_rpm`, `gear`
- `lat_g`, `lon_g`, `yaw_rate_deg_s`
- `current_lap_time_s`

## Suggested Next Steps (for your portfolio)
- **Corner & sector overlays** (shade plots for T1–T12, Sector 1–3)
- **Braking zone detection** (threshold on `brake_pct` and `lon_g` ≤ −0.4 g)
- **Minimum corner speed table** (report min speed per corner)
- **Driver comparison overlay** (Lap A vs Lap B traces on the same axes)
- **Export PDF report** with key metrics

## Notes
- Data are synthetic but shaped to look IndyCar‑like on a 3.6 km road course with ~90s lap.
- Later you can swap `--data` to a real iRacing/AC export once you have one.

Happy building!
