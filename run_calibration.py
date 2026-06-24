"""
run_calibration.py — HH3/IS3 波长定标（锚点匹配模式）

职责：选择数据文件 → 判定光源类型 → 全谱寻峰 → 锚点匹配 → 多项式拟合 → 导出

支持两种定标模式：
    单灯模式：选择一个 HgAr 灯文件
    三灯联合模式：选择 KR + AR + NM 三个灯文件

使用方式：
    python run_calibration.py
"""

import os
import re
import sys
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from wavecal import (
    run_explorer, Config,
    calibrate, pixel_to_wavelength, print_calibration_report,
    match_anchors_to_peaks, verify_anchor_ratios,
    AnchorMatchError, AnchorRatioError,
    auto_tune_config, AutoTuneError,
)
from lamp_registry import get_lamp_config, detect_lamp_from_filename


# ==============================================================================
# 全局配置
# ==============================================================================
CALIBRATION_DEGREE = 3
ANCHOR_MATCH_TOLERANCE = 5.0     # 锚点匹配容差（像素）
ANCHOR_RATIO_TOLERANCE = 0.10    # 比例验证容差


# ==============================================================================
# 文件读取
# ==============================================================================

def _detect_skiprows(path: str) -> int:
    """检测数据起始行：跳过 Date/Temperature/Wavelength 等元数据行。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 20:
                return 0
            if re.search(r"(date|temperature|weavelenth|weavelength|index)", line, re.I):
                continue
            parts = [p for p in re.split(r"[,\t ]+", line.strip()) if p]
            if len(parts) >= 2:
                num_pat = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
                if sum(1 for p in parts if num_pat.match(p)) / len(parts) >= 0.8:
                    return i
    return 0


def read_spectrum(path: str):
    """
    读取光谱文件，返回 (intensity, slope_sign)。
    slope_sign: +1=波长随像素递增, -1=递减(传感器倒置)
    """
    skiprows = _detect_skiprows(path)
    df = pd.read_csv(path, header=None, skiprows=skiprows)

    if df.shape[1] >= 2:
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        intensity = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        intensity = col0.copy()

    valid = ~(col0.isna() | intensity.isna())
    col0_vals = np.asarray(col0[valid], dtype=float)
    intensity = np.asarray(intensity[valid], dtype=float)

    slope_sign = _detect_slope(col0_vals)
    return intensity, slope_sign


def _detect_slope(col0: np.ndarray) -> int:
    """从 CSV 第一列推断色散方向。+1=递增, -1=递减。"""
    if len(col0) < 2:
        return +1
    if float(col0.max()) > 300:
        return -1 if float(col0[0]) > float(col0[-1]) else +1
    return +1


# ==============================================================================
# 步骤函数
# ==============================================================================

def select_csv_files() -> list:
    """弹出多选文件窗口。"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_paths = filedialog.askopenfilenames(
        title="选择定标灯文件（单灯选1个HgAr，三灯选KR+AR+NM各1个）",
        filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    root.destroy()
    if not file_paths:
        raise SystemExit("用户取消了文件选择。")
    return list(file_paths)


def detect_calibration_mode(file_paths: list) -> tuple:
    """根据文件名和数量判断定标模式。"""
    n = len(file_paths)

    if n == 1:
        fname = os.path.basename(file_paths[0])
        detected = detect_lamp_from_filename(fname)
        if detected == "HgAr":
            return ("single", "HgAr")
        reason = (f"识别为 [{detected}]，但单灯模式仅支持 HgAr"
                  if detected else "文件名中未识别出任何已注册的光源标识")
        _fail_mode_detection(f"选择了 1 个文件，{reason}。\n文件: {fname}")

    if n == 3:
        file_lamp = {}
        for p in file_paths:
            fname = os.path.basename(p).upper()
            matches = [l for l in ("KR", "AR", "NM") if l in fname]
            file_lamp[p] = matches[0] if len(matches) == 1 else None
        matched = [v for v in file_lamp.values() if v in ("KR", "AR", "NM")]
        if sorted(matched) == ["AR", "KR", "NM"]:
            return ("multi", ["KR", "AR", "NM"])
        _fail_mode_detection(f"选择了 3 个文件，但不能唯一对应 KR/AR/NM 三灯。")

    _fail_mode_detection(f"选择了 {n} 个文件，无法判断定标模式。")


def _fail_mode_detection(reason: str):
    messagebox.showerror("无法判断定标模式", reason)
    raise SystemExit(reason)


def run_full_pipeline(intensity: np.ndarray, log_path: str):
    cfg = Config(log_path=log_path, quality_csv_path="")
    return run_explorer(intensity, cfg, save_csv=False)


def generate_report(report_path, file_path, lamp_name, lamp_description,
                    matched_pairs, calibration_result):
    """生成定标质量报告。"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pairs_sorted = sorted(matched_pairs, key=lambda x: x[0])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("光谱仪波长定标报告 (Wavelength Calibration Report)\n")
        f.write("=" * 60 + "\n")
        f.write(f"定标处理时间: {current_time}\n")
        f.write(f"数据源输入文件: {file_path}\n")
        f.write(f"选定光源类型: {lamp_name} ({lamp_description})\n")
        f.write(f"识别方式: 锚点匹配（仪器专属锚点像素位置）\n")
        f.write(f"匹配点数: {len(pairs_sorted)}\n\n")

        f.write("------------------ 寻峰与拟合明细 ------------------\n")
        f.write(f"{'序号':<4}{'标准波长(nm)':<14}"
                f"{'质心像素':<14}{'标定波长(nm)':<16}{'残差(nm)':<12}\n")

        for i, (px, wl) in enumerate(pairs_sorted):
            fitted_wl = calibration_result.fitted_wavelengths[i]
            resid = calibration_result.residuals_nm[i]
            f.write(f"{i+1:<6}{wl:<16.3f}{px:<16.3f}"
                    f"{fitted_wl:<18.4f}{resid:+.4f}\n")

        f.write("\n------------------ 拟合质量评估 ------------------\n")
        f.write(f"多项式拟合阶数: {calibration_result.degree} 阶\n")
        f.write(f"最大绝对残差: {calibration_result.max_resid_nm:.4f} nm\n")
        f.write(f"定标均方根残差 (RMSE): {calibration_result.rms_nm:.4f} nm "
                f"({calibration_result.rms_px:.4f} px)\n")
        f.write(f"中心像素色散率: {calibration_result.dispersion_nm_per_px:.4f} nm/px\n\n")

        f.write(f"拟合多项式系数 (degree={calibration_result.degree}):\n")
        for i, c in enumerate(calibration_result.coeffs):
            power = calibration_result.degree - i
            f.write(f"c[{power}] = {c:.8e}\n")
        f.write("=" * 60 + "\n")


# ==============================================================================
# 单灯锚点定标
# ==============================================================================

def run_single_lamp_anchor(
    file_path: str, lamp_name: str, lamp,
    intensity: np.ndarray, slope_sign: int,
    dir_name: str, base_name: str,
):
    """单灯锚点模式：HgAr，锚点匹配 + 多项式拟合。"""
    N = len(intensity)
    log_path = os.path.join(dir_name, f"{base_name}_peakfind_log.txt")

    # flip
    if slope_sign == -1:
        intensity_proc = intensity[::-1]
        print(f"[INFO] 传感器倒置，flip 数据 (N={N})。")
    else:
        intensity_proc = intensity

    # auto_tune
    base_cfg = Config(log_path=log_path, quality_csv_path="")
    try:
        cfg = auto_tune_config({"HgAr": intensity_proc}, base_config=base_cfg)
    except AutoTuneError as e:
        raise RuntimeError(f"自动调参失败: {e}") from e

    # 寻峰
    pipeline_result = run_full_pipeline(intensity_proc, log_path)
    centroids_proc = pipeline_result.centroids
    if len(centroids_proc) == 0:
        raise RuntimeError(f"全谱寻峰未找到任何候选峰。详见: {log_path}")

    # 锚点匹配
    if lamp.anchor_pixels is None:
        raise RuntimeError(
            f"光源 [{lamp_name}] 未配置 anchor_pixels，无法进行锚点定标。"
        )

    anchor_matches = match_anchors_to_peaks(
        anchor_pixels=lamp.anchor_pixels,
        peak_centroids=centroids_proc,
        match_tolerance=ANCHOR_MATCH_TOLERANCE,
    )
    verify_anchor_ratios(anchor_matches, tolerance=ANCHOR_RATIO_TOLERANCE)

    # 构建 (px, wl) 对
    pairs_proc = [(m.matched_centroid, lamp.true_wavelengths[m.anchor_index])
                  for m in anchor_matches]

    # unflip
    if slope_sign == -1:
        pairs = [(N - 1 - px, wl) for px, wl in pairs_proc]
    else:
        pairs = pairs_proc

    # 拟合 + 导出
    calibration_result = calibrate(
        centroids=[p[0] for p in pairs],
        wavelengths=[p[1] for p in pairs],
        degree=CALIBRATION_DEGREE,
    )
    print_calibration_report(calibration_result, matches=anchor_matches)

    all_pixels = np.arange(N)
    all_wavelengths = pixel_to_wavelength(all_pixels, calibration_result)
    out_path = os.path.join(dir_name, f"{base_name}_calibrated.csv")
    pd.DataFrame({"Wavelength_nm": all_wavelengths, "Intensity": intensity}).to_csv(
        out_path, index=False, header=False
    )

    report_path = os.path.join(dir_name, f"{base_name}_wavecal_report.txt")
    generate_report(report_path, file_path, lamp_name, lamp.description,
                    pairs, calibration_result)

    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    messagebox.showinfo(
        "波长定标成功",
        f"定标任务顺利完成！\n\n"
        f"光源类型: {lamp_name}\n"
        f"锚点匹配数: {len(pairs)} / {len(lamp.true_wavelengths)}\n"
        f"RMS残差: {calibration_result.rms_nm:.4f} nm\n"
        f"最大残差: {calibration_result.max_resid_nm:.4f} nm\n\n"
        f"已导出:\n"
        f"1. {base_name}_calibrated.csv\n"
        f"2. {base_name}_wavecal_report.txt\n"
        f"3. {base_name}_peakfind_log.txt",
    )
    root.destroy()
    print(f"定标成功完成，结果已保存至: {dir_name}")


# ==============================================================================
# 三灯锚点定标
# ==============================================================================

MULTI_LAMP_WAVELENGTHS = [
    435.833, 546.074, 587.092, 696.543, 727.294,
    785.482, 850.887, 866.794, 892.869, 965.779, 1013.976,
]


def run_multi_lamp_anchor(
    file_paths: list, lamp_names: list, dir_name: str,
):
    """三灯锚点模式：KR+AR+NM，各自锚点匹配后联合拟合。"""
    lamp_intensities_orig = {}
    lamp_intensities_proc = {}
    lamp_files = {}
    slope_sign = 0
    N = 0

    for p in file_paths:
        fname = os.path.basename(p).upper()
        for lamp in ["KR", "AR", "NM"]:
            if lamp in fname and lamp not in lamp_intensities_orig:
                intensity, ss = read_spectrum(p)
                lamp_intensities_orig[lamp] = intensity
                lamp_files[lamp] = p
                if slope_sign == 0:
                    slope_sign = ss
                    N = len(intensity)
                break

    if len(lamp_intensities_orig) != 3:
        raise RuntimeError(
            f"需要 KR/AR/NM 各一个文件，当前: {list(lamp_intensities_orig.keys())}"
        )

    # flip
    if slope_sign == -1:
        for lamp in lamp_intensities_orig:
            lamp_intensities_proc[lamp] = lamp_intensities_orig[lamp][::-1]
        print(f"[INFO] 传感器倒置，flip 数据 (N={N})。")
    else:
        lamp_intensities_proc = dict(lamp_intensities_orig)

    # auto_tune
    base_cfg = Config(
        log_path=os.path.join(dir_name, "multi_lamp_peakfind_log.txt"),
        quality_csv_path="",
    )
    try:
        cfg = auto_tune_config(lamp_intensities_proc, base_config=base_cfg)
    except AutoTuneError as e:
        raise RuntimeError(f"自动调参失败: {e}") from e

    # 各灯寻峰 + 锚点匹配
    all_pairs_proc = []
    for lamp_name in ["KR", "AR", "NM"]:
        lamp = get_lamp_config(lamp_name)
        intensity = lamp_intensities_proc[lamp_name]

        log_path = os.path.join(dir_name, f"multi_lamp_{lamp_name}_peakfind_log.txt")
        pl = run_full_pipeline(intensity, log_path)
        centroids = pl.centroids
        if len(centroids) == 0:
            raise RuntimeError(f"[{lamp_name}] 未找到候选峰。")

        if lamp.anchor_pixels is None:
            raise RuntimeError(f"[{lamp_name}] 未配置 anchor_pixels。")

        matches = match_anchors_to_peaks(
            anchor_pixels=lamp.anchor_pixels,
            peak_centroids=centroids,
            match_tolerance=ANCHOR_MATCH_TOLERANCE,
        )
        verify_anchor_ratios(matches, tolerance=ANCHOR_RATIO_TOLERANCE)

        for m in matches:
            all_pairs_proc.append(
                (m.matched_centroid, lamp.true_wavelengths[m.anchor_index])
            )

    # unflip
    if slope_sign == -1:
        all_pairs = [(N - 1 - px, wl) for px, wl in all_pairs_proc]
    else:
        all_pairs = all_pairs_proc

    # 拟合
    calibration_result = calibrate(
        centroids=[p[0] for p in all_pairs],
        wavelengths=[p[1] for p in all_pairs],
        degree=CALIBRATION_DEGREE,
    )
    print_calibration_report(calibration_result)

    # 导出每个灯的 CSV（原始顺序）
    for lamp_name, intensity in lamp_intensities_orig.items():
        all_px = np.arange(len(intensity))
        all_wls = pixel_to_wavelength(all_px, calibration_result)
        out_path = os.path.join(dir_name, f"multi_lamp_{lamp_name}_calibrated.csv")
        pd.DataFrame({"Wavelength_nm": all_wls, "Intensity": intensity}).to_csv(
            out_path, index=False, header=False
        )

    # 报告
    report_path = os.path.join(dir_name, "multi_lamp_wavecal_report.txt")
    file_list_str = "; ".join(lamp_files.values())
    generate_report(report_path, file_list_str, "KR+AR+NM",
                    "三灯联合定标", all_pairs, calibration_result)

    n_matched = len(all_pairs)
    n_total = len(MULTI_LAMP_WAVELENGTHS)
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    messagebox.showinfo(
        "三灯联合定标成功",
        f"定标任务顺利完成！\n\n"
        f"定标模式: KR + AR + NM 三灯锚点联合\n"
        f"匹配点数: {n_matched} / {n_total}\n"
        f"RMS残差: {calibration_result.rms_nm:.4f} nm\n"
        f"最大残差: {calibration_result.max_resid_nm:.4f} nm\n\n"
        f"已导出 per-lamp _calibrated.csv 和定标报告",
    )
    root.destroy()
    print(f"三灯联合定标完成，结果已保存至: {dir_name}")


# ==============================================================================
# 主流程
# ==============================================================================

def run_wavelength_calibration():
    try:
        file_paths = select_csv_files()
        first_path = file_paths[0]
        dir_name = os.path.dirname(first_path)

        mode, info = detect_calibration_mode(file_paths)

        if mode == "single":
            lamp_name = info
            lamp = get_lamp_config(lamp_name)
            intensity, slope_sign = read_spectrum(first_path)
            base_name = os.path.splitext(os.path.basename(first_path))[0]
            run_single_lamp_anchor(
                first_path, lamp_name, lamp, intensity, slope_sign,
                dir_name, base_name,
            )

        elif mode == "multi":
            run_multi_lamp_anchor(file_paths, info, dir_name)

    except (AnchorMatchError, AnchorRatioError) as e:
        messagebox.showerror("定标中止：锚点匹配失败", str(e))
        print(f"定标中止: {e}", file=sys.stderr)

    except SystemExit as e:
        print(f"已取消: {e}")

    except Exception as e:
        messagebox.showerror("定标中止：未预期的错误", str(e))
        print(f"定标中止（未预期错误）: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    run_wavelength_calibration()
