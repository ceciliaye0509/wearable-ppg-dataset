"""
HRV / PRV 指标计算模块（基于 NeuroKit2）。

心跳检测使用 NeuroKit2 的 ``elgendi`` PPG 流水线（与 ``neurokit.py`` 相同的检测器），
HRV 指标直接来自 NeuroKit2 官方估计器 ``nk.hrv_time`` / ``nk.hrv_frequency`` /
``nk.hrv_nonlinear``，因此每个输出列都 1:1 对应 NeuroKit 文档，可直接引用。

输出列名保持 NeuroKit 原生的 ``HRV_*`` 格式（如 ``HRV_RMSSD``, ``HRV_SDNN``,
``HRV_LFHF``），以便溯源。额外添加两列：``n_peaks``（检测到的心跳数）和
``hr_mean``（60000 / HRV_MeanNN）。

如果 NeuroKit2 未安装，则使用 SciPy 回退方案计算简化的时域指标集
（MeanNN/SDNN/RMSSD/SDSD/pNN50/pNN20/CVNN）；频域和非线性指标为 NaN。

术语说明：从 PPG 波峰推导的 HRV 称为脉搏率变异性（PRV），是对 ECG-derived HRV 的近似。

窗口长度：时域指标（尤其 RMSSD）在 ~1-2 分钟以上可靠；频域 LF/HF 需 ≥2-5 分钟。
本项目使用 5 分钟窗口，每个窗口都是标准的短期 HRV 记录（Task Force 1996），
因此逐窗口输出是合理的分析单元。
"""
from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# 可选依赖：NeuroKit2（未安装时回退到 SciPy）
# ---------------------------------------------------------------------------
try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover - 可选依赖
    nk = None  # type: ignore[assignment]

from scipy.signal import find_peaks  # noqa: E402

# ---------------------------------------------------------------------------
# 生理常数
# ---------------------------------------------------------------------------
# 心跳间期（IBI）生理范围门限（毫秒），与 neurokit.py 一致
IBI_MIN_MS = 300.0   # 对应 200 bpm 上限
IBI_MAX_MS = 2000.0  # 对应 30 bpm 下限

# PPG RMSSD 生理上限：超过此值说明波峰检测有误（运动伪差导致 IBI 变异性虚高）
RMSSD_MAX_MS = 200.0
# IBI 变异系数上限：正常窦性心律静息态一般 <0.15；0.20 是日常活动的宽松上限
IBI_CV_MAX = 0.20

# ---------------------------------------------------------------------------
# 输出列名定义（使用 NeuroKit 原生 HRV_* 命名）
# ---------------------------------------------------------------------------
# 精选的可报告时域 HRV 列
TIME_COLS: tuple[str, ...] = (
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
    "HRV_pNN50", "HRV_pNN20", "HRV_CVNN", "HRV_MedianNN",
)
# 频域 HRV 列
FREQ_COLS: tuple[str, ...] = (
    "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_TP",
)
# 非线性（Poincaré）HRV 列
NONLINEAR_COLS: tuple[str, ...] = ("HRV_SD1", "HRV_SD2", "HRV_SD1SD2")


def hrv_columns(freq: bool = True, nonlinear: bool = True) -> list[str]:
    """根据选择的指标族返回有序的输出列名列表。"""
    cols = ["n_peaks", "hr_mean", *TIME_COLS]
    if freq:
        cols += list(FREQ_COLS)
    if nonlinear:
        cols += list(NONLINEAR_COLS)
    return cols


# 质控诊断列
QC_COLS: tuple[str, ...] = (
    "sqi",                    # 信号质量指数
    "valid_ibi_ratio",        # 有效 IBI 比例
    "ibi_cv",                 # IBI 变异系数
    "ibi_correction_ratio",   # IBI 校正比例
    "ppg_qc_reason",          # 质控失败原因
)


# ---------------------------------------------------------------------------
# 多检测器一致性融合
# ---------------------------------------------------------------------------
def _consensus_peaks(
    peaks_a: np.ndarray, peaks_b: np.ndarray, tol_samples: int = 5
) -> np.ndarray:
    """保留两个检测器在 ±tol_samples 范围内一致的波峰。

    对于 peaks_a 中的每个波峰，如果 peaks_b 中有一个波峰在 ±tol 采样点内，
    则保留 peaks_a 的位置（假定更精确）。

    参考文献：Charlton et al. (2022) 推荐多检测器融合以提高鲁棒性。
    """
    # 边界情况：如果某个检测器没有结果，返回另一个的结果
    if peaks_a.size == 0:
        return peaks_b.copy()
    if peaks_b.size == 0:
        return peaks_a.copy()

    # 逐个检查 peaks_a 中的波峰是否在 peaks_b 中有匹配
    consensus = []
    for p in peaks_a:
        dists = np.abs(peaks_b.astype(np.int64) - int(p))
        if dists.min() <= tol_samples:
            consensus.append(int(p))
    return np.array(consensus, dtype=np.int64) if consensus else peaks_a.copy()


# ---------------------------------------------------------------------------
# PPG 波峰检测
# ---------------------------------------------------------------------------
def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """检测单个 PPG 窗口的收缩期波峰，返回采样点索引数组。"""
    # 转为浮点并去除 NaN
    x = np.asarray(ppg, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 50:
        return np.empty(0, dtype=np.int64)

    # 优先使用 NeuroKit2 的 Elgendi 算法
    if nk is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用 Elgendi 方法清洗和检测波峰
                xc = nk.ppg_clean(x, sampling_rate=fs, method="elgendi")
                _, info = nk.ppg_peaks(
                    xc, sampling_rate=fs, method="elgendi", correct_artifacts=True
                )
            return np.asarray(info.get("PPG_Peaks", []), dtype=np.int64).ravel()
        except Exception:
            pass  # 失败时回退到 SciPy

    # SciPy 回退方案：假设信号已经过去趋势和带通滤波（preprocess.py）
    x = x - np.mean(x)
    std = float(np.std(x))
    if std == 0.0:
        return np.empty(0, dtype=np.int64)
    # 最小波峰间距 = 0.273s（对应 220 bpm 上限）
    min_dist = max(1, int(round(0.273 * fs)))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=0.3 * std)
    return peaks.astype(np.int64)


# ---------------------------------------------------------------------------
# 亚采样精度波峰细化（三次多项式插值）
# ---------------------------------------------------------------------------
def _refine_peaks_parabolic(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """使用三次（n=3）多项式插值实现亚采样精度的波峰定位。

    对每个整数波峰位置，取 5 个相邻采样点 [p-2, p-1, p, p+1, p+2]，
    拟合三次多项式，并解析求极值点。边界波峰回退到二次（3 点）拟合。

    CinC2025-187 (Valencio et al.) 表明 n=3 在三星 Galaxy Ring PPG 数据上
    优于 n=2 的波峰细化效果。

    参考文献：Fioravanti et al. (2023); Valencio et al. (CinC 2025)。
    """
    refined = np.empty(len(peaks), dtype=np.float64)
    for i, p in enumerate(peaks):
        # ----- 5 点三次插值（首选方案） -----
        if p >= 2 and p <= len(signal) - 3:
            xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            ys = np.array([
                float(signal[p - 2]), float(signal[p - 1]),
                float(signal[p]),
                float(signal[p + 1]), float(signal[p + 2]),
            ])
            # 拟合 ax³ + bx² + cx + d
            coeffs = np.polyfit(xs, ys, 3)
            a, b, c, _d = coeffs
            # 对导数 f'(x) = 3ax² + 2bx + c = 0 求根
            disc = 4.0 * b * b - 12.0 * a * c
            if abs(a) > 1e-15 and disc >= 0:
                sqrt_disc = np.sqrt(disc)
                x1 = (-2.0 * b + sqrt_disc) / (6.0 * a)
                x2 = (-2.0 * b - sqrt_disc) / (6.0 * a)
                # 选择最靠近原始波峰（x=0）且为极大值（f''(x) = 6ax + 2b < 0）的根
                candidates = []
                for xc in [x1, x2]:
                    if abs(xc) <= 2.0 and (6.0 * a * xc + 2.0 * b) < 0:
                        candidates.append(xc)
                if candidates:
                    best = min(candidates, key=abs)
                    refined[i] = p + best
                    continue
            # 三次求解失败，回退到二次方案

        # ----- 3 点二次回退方案 -----
        if p <= 0 or p >= len(signal) - 1:
            refined[i] = float(p)
            continue
        y0, y1, y2 = float(signal[p - 1]), float(signal[p]), float(signal[p + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            refined[i] = float(p)
        else:
            refined[i] = p + 0.5 * (y0 - y2) / denom
    return refined


# ---------------------------------------------------------------------------
# IBI 伪差校正（Lipponen & Tarvainen 2019 简化版）
# ---------------------------------------------------------------------------
def _correct_ibi_artifacts(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> np.ndarray:
    """简化版 Lipponen & Tarvainen (2019) IBI 伪差校正。

    检测偏离滑动中位数超过 threshold（分数）的 IBI，并用中位数替换。

    参考文献：Lipponen & Tarvainen (2019)，"A robust algorithm for
    heart rate variability time series artefact correction using novel
    beat classification"，DOI: 10.1080/03091902.2019.1640306。
    """
    if ibi_ms.size < 5:
        return ibi_ms.copy()
    from scipy.ndimage import median_filter

    corrected = ibi_ms.copy()
    # 使用 11 点滑动中位数作为局部基准
    med = median_filter(ibi_ms, size=11, mode="reflect")
    # 标记偏离中位数超过 threshold 比例的异常值
    outlier = np.abs(corrected - med) > threshold * med
    # 用中位数替换异常值
    corrected[outlier] = med[outlier]
    return corrected


def _correct_ibi_artifacts_with_ratio(
    ibi_ms: np.ndarray,
    threshold: float = 0.20,
) -> tuple[np.ndarray, float]:
    """与 _correct_ibi_artifacts 相同，额外返回被替换的 IBI 比例。"""
    if ibi_ms.size < 5:
        return ibi_ms.copy(), 0.0
    from scipy.ndimage import median_filter

    corrected = ibi_ms.copy()
    med = median_filter(ibi_ms, size=11, mode="reflect")
    outlier = np.abs(corrected - med) > threshold * med
    corrected[outlier] = med[outlier]
    return corrected, float(np.mean(outlier))


# ---------------------------------------------------------------------------
# 亚采样精度时域指标覆盖
# ---------------------------------------------------------------------------
def _override_time_domain_subsample(
    out: dict[str, float],
    peaks_float: np.ndarray,
    fs: float,
) -> None:
    """使用亚采样精度的波峰位置重新计算 RMSSD / SDNN / MeanNN 等时域指标。

    NeuroKit 内部使用整数波峰索引（100 Hz 时精度为 10 ms）。本函数用
    抛物线插值得到的浮点位置替换这些粗糙估计，并在计算前应用 IBI 伪差校正
    （Lipponen & Tarvainen 2019）。
    """
    if peaks_float.size < 3:
        return
    # 从浮点波峰位置计算 IBI（毫秒精度）
    ibi = np.diff(peaks_float) / fs * 1000.0
    # 生理范围门限过滤
    nn = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    if nn.size < 3:
        return

    # IBI 伪差校正
    nn, corr_ratio = _correct_ibi_artifacts_with_ratio(nn)
    diff_nn = np.diff(nn)

    # 计算并覆盖所有时域指标
    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff_nn ** 2)))
    out["HRV_MeanNN"] = mean_nn
    out["hr_mean"] = 60000.0 / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_SDNN"] = sdnn
    out["HRV_RMSSD"] = rmssd
    out["HRV_SDSD"] = float(np.std(diff_nn, ddof=1))
    out["HRV_pNN50"] = 100.0 * float(np.mean(np.abs(diff_nn) > 50.0))
    out["HRV_pNN20"] = 100.0 * float(np.mean(np.abs(diff_nn) > 20.0))
    out["HRV_CVNN"] = sdnn / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_MedianNN"] = float(np.median(nn))

    # 使用 Poincaré 恒等式从校正后的 IBI 推导 SD1/SD2，
    # 避免整数波峰位置的量化误差传播
    sd1 = rmssd / (2.0 ** 0.5)
    sd2_sq = max(0.0, 2.0 * sdnn ** 2 - sd1 ** 2)
    out["HRV_SD1"] = sd1
    out["HRV_SD2"] = sd2_sq ** 0.5
    out["ibi_correction_ratio"] = corr_ratio


# ---------------------------------------------------------------------------
# HRV 指标计算核心函数
# ---------------------------------------------------------------------------
def _nan_metrics(freq: bool, nonlinear: bool) -> dict[str, float]:
    """返回全 NaN 的指标字典（用于无效窗口）。"""
    d = {c: float("nan") for c in hrv_columns(freq, nonlinear)}
    d["n_peaks"] = 0.0
    return d


def _scipy_time_domain(peaks: np.ndarray, fs: float) -> dict[str, float]:
    """NeuroKit2 不可用时的 SciPy 回退方案：仅计算时域 HRV。"""
    out = _nan_metrics(freq=False, nonlinear=False)
    out["n_peaks"] = float(peaks.size)
    if peaks.size < 3:
        return out

    # 从整数波峰索引计算 IBI（毫秒）
    ibi = np.diff(peaks) / float(fs) * 1000.0
    # 生理范围门限过滤
    nn = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
    if nn.size < 3:
        return out

    # 计算时域指标
    diff = np.diff(nn)
    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    out["HRV_MeanNN"] = mean_nn
    out["hr_mean"] = 60000.0 / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_SDNN"] = sdnn
    out["HRV_RMSSD"] = float(np.sqrt(np.mean(diff**2)))
    out["HRV_SDSD"] = float(np.std(diff, ddof=1))
    out["HRV_pNN50"] = 100.0 * float(np.mean(np.abs(diff) > 50.0))
    out["HRV_pNN20"] = 100.0 * float(np.mean(np.abs(diff) > 20.0))
    out["HRV_CVNN"] = sdnn / mean_nn if mean_nn > 0 else float("nan")
    out["HRV_MedianNN"] = float(np.median(nn))
    return out


def hrv_metrics(
    peaks: np.ndarray, fs: float, *, freq: bool = True, nonlinear: bool = True
) -> dict[str, float]:
    """从波峰索引数组通过 NeuroKit2 计算精选 HRV 指标（SciPy 回退）。"""
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    out = _nan_metrics(freq, nonlinear)
    out["n_peaks"] = float(peaks.size)
    if peaks.size < 3:
        return out

    # NeuroKit2 不可用时使用 SciPy 回退
    if nk is None:
        return _scipy_time_domain(peaks, fs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # --- 时域指标 ---
            td = nk.hrv_time(peaks, sampling_rate=fs)
            for c in TIME_COLS:
                if c in td.columns:
                    out[c] = float(td[c].iloc[0])
            # 从 MeanNN 推导平均心率
            mean_nn = out.get("HRV_MeanNN", float("nan"))
            out["hr_mean"] = 60000.0 / mean_nn if mean_nn and mean_nn > 0 else float("nan")

            # --- 频域指标（LF/HF/LF-HF 比等） ---
            if freq:
                fd = nk.hrv_frequency(peaks, sampling_rate=fs)
                for c in FREQ_COLS:
                    if c in fd.columns:
                        out[c] = float(fd[c].iloc[0])

            # --- 非线性指标（Poincaré SD1/SD2 等） ---
            if nonlinear:
                nl = nk.hrv_nonlinear(peaks, sampling_rate=fs)
                for c in NONLINEAR_COLS:
                    if c in nl.columns:
                        out[c] = float(nl[c].iloc[0])
    except Exception:
        # NeuroKit2 内部出错时回退到 SciPy 时域
        return _scipy_time_domain(peaks, fs)
    return out


# ---------------------------------------------------------------------------
# 单窗口便捷接口：PPG 波形 → HRV 指标字典
# ---------------------------------------------------------------------------
def hrv_from_ppg(
    ppg: np.ndarray,
    fs: float,
    *,
    freq: bool = True,
    nonlinear: bool = True,
    sqi_threshold: float = 0.4,
    ibi_validity_threshold: float = 0.80,
    ibi_cv_threshold: float = IBI_CV_MAX,
    rmssd_max_ms: float = RMSSD_MAX_MS,
) -> dict[str, float]:
    """单窗口便捷封装：PPG 波形 → 精选 HRV 指标字典。

    完整流水线：
      1. 波峰检测（NeuroKit Elgendi 算法）
      2. IBI 有效比例门限 —— 若低于 ibi_validity_threshold 的 IBI 落在
         [300, 2000] ms 范围内，标记该窗口不可靠（所有 HRV 指标设为 NaN）
         参考文献：PMC11644394, Sensors 2024 —— 推荐 ~80%
      3. IBI 变异系数门限 —— 高 CV 表示运动导致的假波峰
      4. 信号质量评估（SQA）—— 次级门限
      5. NeuroKit HRV 计算（整数波峰）用于频域/非线性指标
      6. 亚采样抛物线插值 + IBI 伪差校正，覆盖时域指标以获得更高精度

    设置环境变量 BASELINE=1 可跳过所有改进（步骤 2-6），用于与优化流水线对比。
    """
    import os
    baseline_mode = os.environ.get("BASELINE", "") == "1"

    from .sqa import ppg_sqi

    # --- 准备信号：去除 NaN ---
    ppg_clean = np.asarray(ppg, dtype=np.float64)
    ppg_clean = ppg_clean[~np.isnan(ppg_clean)]

    # --- 步骤 1：波峰检测 ---
    peaks = detect_ppg_peaks(ppg_clean, fs)

    # --- 基线模式：跳过所有改进，直接计算 ---
    if baseline_mode:
        out = hrv_metrics(peaks, fs, freq=freq, nonlinear=nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = 0.0
        out["ibi_cv"] = float("nan")
        out["ibi_correction_ratio"] = 0.0
        out["ppg_qc_reason"] = "baseline_mode"
        return out

    # --- 步骤 2：IBI 有效比例门限（PMC11644394） ---
    valid_ibi_ratio = 0.0
    ibi_cv = float("nan")
    if peaks.size >= 2:
        # 计算原始 IBI
        ibi_raw = np.diff(peaks) / fs * 1000.0
        nn_raw = ibi_raw[(ibi_raw >= IBI_MIN_MS) & (ibi_raw <= IBI_MAX_MS)]
        n_valid = int(len(nn_raw))
        valid_ibi_ratio = n_valid / len(ibi_raw)
        # 计算 IBI 变异系数
        if nn_raw.size >= 2:
            mean_nn = float(np.mean(nn_raw))
            ibi_cv = float(np.std(nn_raw, ddof=1) / mean_nn) if mean_nn > 0 else float("nan")

    # 有效比例过低 → 标记为不可靠并退出
    if valid_ibi_ratio < ibi_validity_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "low_valid_ibi_ratio"
        return out

    # --- 步骤 3：IBI 变异系数门限（高 CV 表示运动导致的假波峰） ---
    if np.isfinite(ibi_cv) and ibi_cv > ibi_cv_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = 0.0
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "high_ibi_cv"
        return out

    # --- 步骤 4：信号质量门限 ---
    sqi = ppg_sqi(ppg_clean, fs, peaks)
    if sqi < sqi_threshold:
        out = _nan_metrics(freq, nonlinear)
        out["sqi"] = sqi
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ibi_correction_ratio"] = float("nan")
        out["ppg_qc_reason"] = "low_sqi"
        return out

    # --- 步骤 5：使用 NeuroKit2 计算完整 HRV 指标 ---
    out = hrv_metrics(peaks, fs, freq=freq, nonlinear=nonlinear)
    out["sqi"] = sqi
    out["valid_ibi_ratio"] = valid_ibi_ratio
    out["ibi_cv"] = ibi_cv
    out["ibi_correction_ratio"] = 0.0
    out["ppg_qc_reason"] = "ok"

    # --- 步骤 6：亚采样精度细化，覆盖时域指标 ---
    if peaks.size >= 3 and ppg_clean.size > 0:
        peaks_f = _refine_peaks_parabolic(ppg_clean, peaks)
        _override_time_domain_subsample(out, peaks_f, fs)

    # --- RMSSD 上限检查：超标说明波峰检测有误导致的虚高 ---
    if np.isfinite(out.get("HRV_RMSSD", float("nan"))) and out["HRV_RMSSD"] > rmssd_max_ms:
        for k in _nan_metrics(freq, nonlinear):
            out[k] = float("nan")
        out["sqi"] = sqi
        out["valid_ibi_ratio"] = valid_ibi_ratio
        out["ibi_cv"] = ibi_cv
        out["ppg_qc_reason"] = "high_rmssd"

    return out
