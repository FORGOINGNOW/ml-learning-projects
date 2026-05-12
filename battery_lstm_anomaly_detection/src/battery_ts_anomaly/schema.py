from __future__ import annotations

TEMP_COLS = [f"battery_cell_temp_{idx}" for idx in range(1, 9)]
VOLT_COLS = [f"battery_cell_volt_{idx}" for idx in range(1, 9)]
SOC_COLS = [f"battery_cell_SOC_{idx}" for idx in range(1, 9)]

FEATURE_COLUMNS = [
    "current_a",
    "abs_current_a",
    "current_direction",
    "is_charge",
    "is_discharge",
    "hour_sin",
    "hour_cos",
    "throughput_norm",
    "cell_voltage_mean",
    "cell_voltage_min",
    "cell_voltage_max",
    "cell_voltage_std",
    "cell_voltage_delta",
    "cell_temp_mean",
    "cell_temp_max",
    "cell_temp_std",
    "cell_temp_delta",
    "cell_soc_mean",
    "cell_soc_min",
    "cell_soc_std",
    "cell_soc_delta",
    "d_voltage_mean",
    "d_temp_max",
    "d_soc_mean",
]

TARGET_COLUMNS = [
    "cell_voltage_min",
    "cell_voltage_delta",
    "cell_temp_max",
    "cell_temp_delta",
    "cell_soc_mean",
    "cell_soc_delta",
]

META_COLUMNS = [
    "time",
    "device_id",
    "split",
    "state_label",
    "is_anomaly_point",
    "anomaly_start_time",
    "fault_cell",
]
