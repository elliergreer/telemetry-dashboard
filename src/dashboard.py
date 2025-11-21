# Minimal "MoTeC-lite" telemetry dashboard starter
# Usage:
#   python src/dashboard.py --data data/sample_indycar_telemetry.csv
#
# What it does now:
# - Loads telemetry CSV
# - Plots speed & throttle traces for a selected lap
# - Computes a simple lap delta vs chosen reference lap (WIP placeholder)
#
# Next steps you will add:
# - Corner/sector overlays
# - Braking zone detection
# - Min corner speed table
# - Driver comparison (Lap A vs Lap B)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Approximate IndyCar track corner centers (meters)
TURN_CENTERS = [150, 450, 750, 1050, 1340, 1650, 1950, 2250, 2550, 2850, 3150, 3450]
TURN_HALF_WIDTH = 80 # shading half-width around each corner center


def get_turn_windows(): 
    """Returns list of (turn_id, start_m, end_m) for shading + min corner speed."""
    windows = []
    for i, c in enumerate(TURN_CENTERS, start=1):
        windows.append((i, c - TURN_HALF_WIDTH, c + TURN_HALF_WIDTH))
    return windows

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity
    required = ["time_s","lap","distance_m","speed_mps","throttle_pct","brake_pct","steer_deg","engine_rpm","lat_g","lon_g","current_lap_time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def get_laps(df: pd.DataFrame):
    return sorted(df["lap"].unique())

def compute_lap_delta(df: pd.DataFrame, lap_a: int, lap_b: int) -> pd.DataFrame:
    # Very simple lap delta: resample both laps by distance and diff time
    a = df[df["lap"]==lap_a].copy()
    b = df[df["lap"]==lap_b].copy()
    # Distance grid
    dmin = max(a["distance_m"].min(), b["distance_m"].min())
    dmax = min(a["distance_m"].max(), b["distance_m"].max())
    grid = np.linspace(dmin, dmax, 1000)
    a_t = np.interp(grid, a["distance_m"], a["current_lap_time_s"])
    b_t = np.interp(grid, b["distance_m"], b["current_lap_time_s"])
    delta = b_t - a_t  # positive => Lap B is slower than Lap A at that point
    return pd.DataFrame({"distance_m": grid, "delta_s": delta})

def plot_traces(df: pd.DataFrame, lap_sel: int):
   lap_df = df[df["lap"]==lap_sel].copy()
   turn = get_turn_windows()

   #Speed Plot
   plt.figure()
   plt.title(f"Speed vs Distance (Lap{lap_sel})")
   for tid, a , b in turns:
       plt.axvspan(a, b, alpha = 0.15)
   plt.xlabel("Distance (m)")
   plt.ylabel("Speed (m/s)")
   plt.grid(True)
   plt.show()

   #Throttle/Brake Plot
   plt.figure()
   plt.title(f"Throttle/Brake vs Distance (Lap {lap_sel})")
   plt.plot(lap_df["distance_m"], lap_df["brake_pct"], label = "Throttle (%)")
   plt.plot(lap_df["distance_m"], lap_df["brake_pct"], label="Brake (%)")
   for tid, a, b in turns:
       plt.axvspan(a, b, alpha=0.15)
   plt.xlabel("Distance (m)")
   plt.ylabel ("Percent")
   plt.legend()
   plt.grid(True)
   plt.show()

   #Min corner speed table 
   rows = []
   for tid, a,b in turns:
       seg=lap_df[(lap_df["distance_m"] >= a) & (lap_df["distance_m"] <=b)]
       if len(seg):
              rows.append((tid, float(seg["speed_mps"].min())))
   if rows:
       out = pd.DataFrame(rows, colums=["Turn", "MinSpeed_mps"])
       out["MinSpeed_kmh"] = out["MinSpeed_mps"] * 3.6
       print("\nMin corner speeds (Lap {}):".format(lap_sel))
       print(out.to_string(
           index=False,
           formatters={ 
               "MinSpeed_mps":"{:2f}".format,
               "MinSpeed_kmh":"{:.1f}".format;
           }
       ))           

def plot_lap_delta(df: pd.DataFrame, lap_ref: int, lap_cmp: int):
    d = compute_lap_delta(df, lap_ref, lap_cmp)
    plt.figure()
    plt.title(f"Lap Delta (Lap {lap_cmp} vs Lap {lap_ref})")
    plt.plot(d["distance_m"], d["delta_s"])
    for tid, a, b in get_turn_windows():
        plt.axvspan(a, b, alpha=0.15)
    plt.xlabel("Distance (m)")
    plt.ylabel("Delta time (s)  (+ = lap_cmp slower)")
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample_indycar_telemetry.csv")
    parser.add_argument("--lap", type=int, default=1, help="Lap to plot")
    parser.add_argument("--ref", type=int, default=1, help="Reference lap for delta")
    parser.add_argument("--cmp", type=int, default=2, help="Comparison lap for delta")
    args = parser.parse_args()

    df = load_data(args.data)
    laps = get_laps(df)
    if args.lap not in laps:
        args.lap = laps[0]
    if args.ref not in laps:
        args.ref = laps[0]
    if args.cmp not in laps:
        args.cmp = laps[-1]

    print(f"Loaded {len(df)} samples across laps: {laps}")
    plot_traces(df, args.lap)
    if args.ref != args.cmp:
        plot_lap_delta(df, args.ref, args.cmp)

if __name__ == "__main__":
    main()
