"""
HRV / PRV 批量计算器（基于 NeuroKit2）。

与 ``runner.py`` 类似，但不是每个窗口输出一个 HR 标量，而是计算完整的
NeuroKit2 HRV 指标集。复用 ``config.py``、``io_utils.py`` 以及
``runner.py`` / ``preprocess.py`` 生成的 ``*_preprocess.npz`` 缓存。

每个 5 分钟窗口都是标准的短期 HRV 记录，因此逐窗口计算是合理的分析单元。
输出路径为 ``outputs/<Px>/``::

    hrv_<dev>_<channel>.csv   # 每行一个窗口：时域 + 频域 + Poincaré 指标

使用方法::

    python hrv_runner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 包路径设置与导入
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402

from algorithms import hrv  # noqa: E402
from io_utils import (  # noqa: E402
    merged_windows_npz,
    normalize_participant_id,
    participant_device_id,
    ppg_from_npz,
)
from preprocess import write_preprocess_npz  # noqa: E402

# ---------------------------------------------------------------------------
# 从配置中读取计算选项
# ---------------------------------------------------------------------------
HRV_COMPUTE_FREQ: bool = bool(getattr(config, "HRV_COMPUTE_FREQ", True))         # 是否计算频域指标
HRV_COMPUTE_NONLINEAR: bool = bool(getattr(config, "HRV_COMPUTE_NONLINEAR", True))  # 是否计算非线性指标
MOTION_QC_ENABLED: bool = bool(getattr(config, "MOTION_QC_ENABLED", True))        # 是否启用运动质控
MOTION_SEG_SEC: float = float(getattr(config, "MOTION_SEG_SEC", 10.0))            # 运动分析的片段长度（秒）
MOTION_PERCENTILE: float = float(getattr(config, "MOTION_PERCENTILE", 75.0))      # 运动阈值百分位
MOTION_MAX_FRACTION: float = float(getattr(config, "MOTION_MAX_FRACTION", 0.50))  # 高运动片段比例上限

# Step 9：Earring 信号已足够干净，跳过带通滤波（带通反而降低精度）
_SKIP_BANDPASS_ROLES = {"Earring"}


# ---------------------------------------------------------------------------
# 运动质控：基于加速度计的逐窗口运动占比
# ---------------------------------------------------------------------------
def _motion_fraction_per_window(
    data: dict,
    *,
    seg_sec: float = MOTION_SEG_SEC,
    percentile: float = MOTION_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """基于加速度计计算每个窗口的高运动片段占比。

    将每个窗口按 seg_sec 切成小段，计算每段加速度幅值的标准差，
    然后以全局 percentile 分位数为阈值，统计超标片段的比例。

    返回：(每窗口运动占比数组, 运动阈值)
    """
    # 检查加速度计数据是否存在
    if not all(k in data for k in ("accel_x", "accel_y", "accel_z")):
        n = len(np.asarray(data.get("t0_ms", [])))
        return np.full(n, np.nan), float("nan")

    ax = np.asarray(data["accel_x"], dtype=np.float64)
    ay = np.asarray(data["accel_y"], dtype=np.float64)
    az = np.asarray(data["accel_z"], dtype=np.float64)
    fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0
    seg_n = max(1, int(round(seg_sec * fs)))  # 每段的采样点数

    # 第一遍：收集所有窗口所有片段的加速度标准差
    all_stds: list[float] = []
    per_win: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        # 计算三轴加速度合成幅值
        mag = np.sqrt(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2)
        n_segs = len(mag) // seg_n
        if n_segs == 0:
            stds = np.array([float(np.std(mag))])
        else:
            stds = np.array([
                float(np.std(mag[j * seg_n:(j + 1) * seg_n]))
                for j in range(n_segs)
            ])
        per_win.append(stds)
        all_stds.extend(stds.tolist())

    if not all_stds:
        return np.zeros(ax.shape[0]), float("nan")

    # 使用全局 percentile 分位数作为运动阈值
    threshold = float(np.percentile(all_stds, percentile))
    # 每个窗口的高运动片段占比
    fractions = np.array([float(np.mean(stds > threshold)) for stds in per_win])
    return fractions, threshold


# ---------------------------------------------------------------------------
# HRV 指标失效标记
# ---------------------------------------------------------------------------
def _invalidate_hrv_metrics(row: dict, *, reason: str) -> None:
    """将所有 HRV 指标设为 NaN，同时保留质控诊断信息。"""
    for col in hrv.hrv_columns(HRV_COMPUTE_FREQ, HRV_COMPUTE_NONLINEAR):
        if col != "n_peaks":
            row[col] = float("nan")
    # 追加失效原因（可能叠加多个原因）
    existing = str(row.get("ppg_qc_reason", "ok"))
    row["ppg_qc_reason"] = reason if existing == "ok" else f"{existing};{reason}"


# ---------------------------------------------------------------------------
# 数据加载：选择原始或预处理后的 NPZ
# ---------------------------------------------------------------------------
def _load_windows(raw_npz: Path, result_dir: Path, run_preprocess: bool, role: str = "") -> dict:
    """加载窗口数据，根据配置决定是否预处理。

    - 非 baseline 模式下，Earring 跳过带通滤波（Step 9）
    - 其他设备使用预处理后的信号
    """
    import os
    baseline_mode = os.environ.get("BASELINE", "") == "1"

    # Step 9：非 baseline 模式下 Earring 跳过带通
    skip_bp = (not baseline_mode) and (role in _SKIP_BANDPASS_ROLES)
    pre_npz = result_dir / f"{raw_npz.stem}_preprocess.npz"

    if run_preprocess and not skip_bp:
        # 如果预处理文件不存在则生成
        if not pre_npz.is_file():
            print(f"  [预处理] 写入 {pre_npz.name} -> {result_dir}")
            write_preprocess_npz(raw_npz, pre_npz)
        load_path = pre_npz
    else:
        if skip_bp:
            print(f"  [Step 9] {role} 跳过带通滤波 — 使用原始信号")
        load_path = raw_npz

    # 加载并转为字典
    with np.load(load_path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# 单设备单通道 HRV 计算
# ---------------------------------------------------------------------------
def run_one_device_channel(
    *, raw_npz: Path, result_dir: Path, device_id: str, ppg_channel: str,
    run_preprocess: bool, role: str = ""
) -> None:
    """对一个设备的一个 PPG 通道，逐窗口计算 HRV 指标并保存为 CSV。"""
    # 加载数据
    data = _load_windows(raw_npz, result_dir, run_preprocess, role=role)
    t0_ms = np.asarray(data.get("t0_ms", []), dtype=np.float64)
    hr_gt = np.asarray(data.get("hr_gt", []), dtype=np.float64)
    fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0
    ppg = np.asarray(ppg_from_npz(data, ppg_channel), dtype=np.float64)
    n = int(ppg.shape[0])
    result_dir.mkdir(parents=True, exist_ok=True)

    # 计算运动质控信息
    motion_fraction, motion_threshold = _motion_fraction_per_window(data)

    # 逐窗口计算 HRV
    rows = []
    for i in range(n):
        # 使用 hrv_from_ppg 计算完整 HRV 指标（含质控流水线）
        m = hrv.hrv_from_ppg(
            ppg[i], fs, freq=HRV_COMPUTE_FREQ, nonlinear=HRV_COMPUTE_NONLINEAR
        )

        # 组装输出行：窗口起始时间 + ECG ground truth + HRV 指标
        row = {"t0_ms": float(t0_ms[i]) if i < len(t0_ms) else np.nan}
        if i < len(hr_gt):
            row["hr_gt"] = float(hr_gt[i])
        row.update(m)

        # 添加运动质控信息
        mf = float(motion_fraction[i]) if i < len(motion_fraction) else float("nan")
        row["motion_fraction"] = mf
        row["motion_threshold"] = motion_threshold
        # 运动门限判定：高运动片段比例超标则失效
        row["motion_gate"] = bool(
            MOTION_QC_ENABLED
            and np.isfinite(mf)
            and mf >= MOTION_MAX_FRACTION
        )
        if row["motion_gate"]:
            _invalidate_hrv_metrics(row, reason="motion_artifact")
        rows.append(row)

    # 构建 DataFrame 并保存
    cols = ["t0_ms"]
    if len(hr_gt):
        cols.append("hr_gt")
    cols += hrv.hrv_columns(HRV_COMPUTE_FREQ, HRV_COMPUTE_NONLINEAR)
    cols += list(hrv.QC_COLS)
    cols += ["motion_fraction", "motion_threshold", "motion_gate"]
    df = pd.DataFrame(rows, columns=cols)

    out_path = result_dir / f"hrv_{device_id}_{ppg_channel}.csv"
    df.to_csv(out_path, index=False)
    valid = int(np.sum(df["HRV_RMSSD"].notna())) if "HRV_RMSSD" in df else 0
    print(f"  [已保存] {out_path.name}  (有效 RMSSD 窗口数: {valid}/{n})")


# ---------------------------------------------------------------------------
# 主函数：遍历所有参与者/设备/通道
# ---------------------------------------------------------------------------
def main() -> None:
    # 从配置中读取参数
    root = config.HEURISTIC_WINDOWS_ROOT
    participants = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]
    roles = list(config.HEURISTIC_DEVICE_ROLES)
    _ch_cfg = config.HEURISTIC_PPG_CHANNELS
    channels = [_ch_cfg] if isinstance(_ch_cfg, str) else list(_ch_cfg)
    result_root = config.HEURISTIC_RESULT_ROOT

    # 检查数据目录是否存在
    if not root.is_dir():
        print(f"[错误] 窗口 NPZ 根目录不存在: {root}")
        print('  请设置 HEURISTIC_DATA_SOURCE 为 "full" 或 "sample"。')
        sys.exit(1)
    if hrv.nk is None:
        print("[hrv] 警告: neurokit2 未安装 -> 使用 SciPy 时域回退"
              "（无频域/非线性指标）。安装命令: pip install neurokit2")

    # 打印运行配置
    print(f"[hrv] data_source={config.HEURISTIC_DATA_SOURCE!r} windows_npz_root={root}")
    print(f"[hrv] participants={participants} roles={roles} channels={channels}")
    print(
        f"[hrv] preprocess={config.HEURISTIC_RUN_PREPROCESS} "
        f"freq={HRV_COMPUTE_FREQ} nonlinear={HRV_COMPUTE_NONLINEAR} "
        f"backend={'neurokit2' if hrv.nk is not None else 'scipy-fallback'}"
    )

    # 遍历所有参与者 × 设备 × 通道
    for pid in participants:
        result_dir = (result_root / pid).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        for role in roles:
            raw = merged_windows_npz(root, pid, role)
            dev_id = participant_device_id(pid, role)
            if not raw.is_file():
                print(f"\n[跳过] 文件不存在 {raw}")
                continue
            print(f"\n{'='*60}\n  {dev_id}\n  npz={raw.name}\n  输出 -> {result_dir}\n{'='*60}")
            for ch in channels:
                if ch not in ("ppg_ir", "ppg_green"):
                    print(f"  [跳过] 未知通道 {ch}")
                    continue
                print(f"\n  --- 通道 {ch} ---")
                run_one_device_channel(
                    raw_npz=raw,
                    result_dir=result_dir,
                    device_id=dev_id,
                    ppg_channel=ch,
                    run_preprocess=config.HEURISTIC_RUN_PREPROCESS,
                    role=role,
                )

    print("\n[hrv] 完成。")


if __name__ == "__main__":
    main()
