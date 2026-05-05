"""字段定义模块：维护 8 个 cell 的温度、电压、SOC 等标准列名。"""

CELL_COUNT = 8

TEMP_COLS = [f"battery_cell_temp_{idx}" for idx in range(1, CELL_COUNT + 1)]
VOLT_COLS = [f"battery_cell_volt_{idx}" for idx in range(1, CELL_COUNT + 1)]
SOC_COLS = [f"battery_cell_SOC_{idx}" for idx in range(1, CELL_COUNT + 1)]


def sensor_columns() -> list[str]:
    return TEMP_COLS + VOLT_COLS + SOC_COLS + ["current"]


def label_columns() -> list[str]:
    return ["split", "state_label"]
