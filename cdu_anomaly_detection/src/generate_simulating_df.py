import numpy as np
import pandas as pd


def save_dataframe_with_fallback(df: pd.DataFrame, parquet_path: str, csv_path: str) -> str:
    """
    Save dataframe to parquet when engine is available, otherwise save as CSV.
    Returns the saved file path.
    """

    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        print(f"Parquet save failed ({exc}). Falling back to CSV...")
        df.to_csv(csv_path, index=False)
        return csv_path


def generate_cdu_data(
    n_rows: int = 2_000_000,
    n_abnormal: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic cdu_data DataFrame with:
    - classification columns: a, b, c
    - numeric columns:
      CPU_max_temp, CPU_max_power,
      npu_temp_1..8, npu_power_1..8,
      CDU_temp_in, CDU_temp_out, CDU_power_delta
    - device_id column
    - exactly n_abnormal abnormal devices (rows), distributed as evenly as possible
      across combinations of (a, b, c).
    """

    rng = np.random.default_rng(seed)

    # 1) Define classification types (3-5 types each as requested)
    a_types = np.array(["a1", "a2", "a3", "a4"])                  # 4 types
    b_types = np.array(["b1", "b2", "b3", "b4", "b5"])            # 5 types
    c_types = np.array(["c1", "c2", "c3"])                        # 3 types

    # All possible classification combinations
    combos = pd.MultiIndex.from_product(
        [a_types, b_types, c_types], names=["a", "b", "c"]
    ).to_frame(index=False)
    n_combos = len(combos)

    # 2) Build classification columns for n_rows (roughly balanced over combos)
    # Repeat full combo table enough times, then trim
    repeats = (n_rows + n_combos - 1) // n_combos
    cls_df = pd.concat([combos] * repeats, ignore_index=True).iloc[:n_rows].copy()

    # Shuffle classification assignment to avoid block patterns
    perm = rng.permutation(n_rows)
    cls_df = cls_df.iloc[perm].reset_index(drop=True)

    # 3) Generate base "normal" numeric data
    # CPU
    cpu_max_temp = rng.normal(loc=72, scale=7, size=n_rows).clip(45, 95)
    cpu_max_power = rng.normal(loc=145, scale=20, size=n_rows).clip(80, 230)

    # NPU temps/powers (8 chips each)
    npu_temps = []
    npu_powers = []
    for i in range(8):
        t = rng.normal(loc=64 + 0.8 * (i % 3), scale=6.0, size=n_rows).clip(35, 90)
        p = rng.normal(loc=48 + 1.5 * (i % 4), scale=8.0, size=n_rows).clip(20, 95)
        npu_temps.append(t)
        npu_powers.append(p)

    # CDU temps and power delta
    cdu_temp_in = rng.normal(loc=24, scale=2.2, size=n_rows).clip(16, 34)
    # temp_out normally > temp_in by 2~8C
    delta_t = rng.normal(loc=4.5, scale=1.2, size=n_rows).clip(1.5, 9.0)
    cdu_temp_out = (cdu_temp_in + delta_t).clip(18, 45)

    # power_delta roughly related to cooling load
    cdu_power_delta = rng.normal(loc=42, scale=10, size=n_rows).clip(8, 95)

    # 4) Assemble DataFrame
    df = pd.DataFrame({
        "device_id": [f"dev_{i:07d}" for i in range(1, n_rows + 1)],
        "a": cls_df["a"].to_numpy(),
        "b": cls_df["b"].to_numpy(),
        "c": cls_df["c"].to_numpy(),
        "CPU_max_temp": cpu_max_temp,
        "CPU_max_power": cpu_max_power,
        "CDU_temp_in": cdu_temp_in,
        "CDU_temp_out": cdu_temp_out,
        "CDU_power_delta": cdu_power_delta,
    })

    # Add npu columns
    for i in range(8):
        df[f"npu_temp_{i+1}"] = npu_temps[i]
        df[f"npu_power_{i+1}"] = npu_powers[i]

    # 5) Choose abnormal devices, evenly across classification combinations
    # Group row indices by combo
    grouped = df.groupby(["a", "b", "c"], sort=False).indices
    combo_keys = list(grouped.keys())

    base = n_abnormal // n_combos
    rem = n_abnormal % n_combos

    # Randomize which combos get one extra abnormal row
    extra_combo_idx = set(rng.choice(n_combos, size=rem, replace=False).tolist())

    abnormal_indices = []
    for i, key in enumerate(combo_keys):
        need = base + (1 if i in extra_combo_idx else 0)
        if need == 0:
            continue
        idx_pool = grouped[key]
        chosen = rng.choice(idx_pool, size=need, replace=False)
        abnormal_indices.extend(chosen.tolist())

    abnormal_indices = np.array(abnormal_indices, dtype=np.int64)
    assert len(abnormal_indices) == n_abnormal

    # 6) Inject abnormal patterns
    # Strongly high CPU temp/power
    df.loc[abnormal_indices, "CPU_max_temp"] = rng.uniform(96, 110, size=n_abnormal)
    df.loc[abnormal_indices, "CPU_max_power"] = rng.uniform(235, 300, size=n_abnormal)

    # Very high NPU temps/powers on random subset of NPUs per abnormal device
    # (2~5 NPUs per abnormal device)
    for row_idx in abnormal_indices:
        n_hot = int(rng.integers(2, 6))
        hot_npu_ids = rng.choice(np.arange(1, 9), size=n_hot, replace=False)
        for npu_id in hot_npu_ids:
            df.at[row_idx, f"npu_temp_{npu_id}"] = float(rng.uniform(92, 115))
            df.at[row_idx, f"npu_power_{npu_id}"] = float(rng.uniform(98, 140))

    # Abnormal CDU behavior:
    # some rows with very high delta/out temp, some rows with impossible out<in
    half = n_abnormal // 2
    idx_1 = abnormal_indices[:half]
    idx_2 = abnormal_indices[half:]

    df.loc[idx_1, "CDU_temp_out"] = df.loc[idx_1, "CDU_temp_in"].to_numpy() + rng.uniform(10, 20, size=len(idx_1))
    df.loc[idx_1, "CDU_power_delta"] = rng.uniform(100, 180, size=len(idx_1))

    # out < in anomaly + negative-ish/near-zero power delta behavior
    # (still keep as numeric realistic "bad data")
    df.loc[idx_2, "CDU_temp_out"] = df.loc[idx_2, "CDU_temp_in"].to_numpy() - rng.uniform(0.2, 4.0, size=len(idx_2))
    df.loc[idx_2, "CDU_power_delta"] = rng.uniform(-15, 5, size=len(idx_2))

    # 7) Add abnormal flag for convenience
    df["is_abnormal"] = 0
    df.loc[abnormal_indices, "is_abnormal"] = 1

    return df


if __name__ == "__main__":
    cdu_data = generate_cdu_data(n_rows=2_000_000, n_abnormal=100, seed=2026)

    # Quick checks
    print("Shape:", cdu_data.shape)
    print("Abnormal count:", int(cdu_data["is_abnormal"].sum()))
    print("\nClassification combo count (top 10):")
    print(cdu_data.groupby(["a", "b", "c"]).size().head(10))

    print("\nAbnormal per combo (non-zero only, top 20):")
    abnormal_dist = cdu_data[cdu_data["is_abnormal"] == 1].groupby(["a", "b", "c"]).size()
    print(abnormal_dist[abnormal_dist > 0].sort_values(ascending=False).head(20))

    saved_path = save_dataframe_with_fallback(
        cdu_data,
        parquet_path="cdu_data.parquet",
        csv_path="cdu_data.csv",
    )
    print(f"\nSaved dataset to: {saved_path}")