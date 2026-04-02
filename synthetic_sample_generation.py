# synth_waterchem_no_cluster_full_print_ks_with_corrmaps_annotated.py
# -*- coding: utf-8 -*-
"""
说明：
- 本脚本用于在不做聚类的情况下，对水化学数据进行拟合并合成新样本；
- 采用“边际分布（经验或KDE） + Copula（高斯或t）”的两步法：
    1) 每个变量单独拟合边际分布（保持各变量分布形状）；
    2) 以Copula建模变量之间的相关结构（保持变量间的依赖关系）；
    3) 从Copula中抽样得到U（0-1分位），通过各边际的PPF映射回原量纲；
    4) 可选做物理约束修复（电荷平衡、TDS近似平衡）；
- 评估：
    * 边际层面：逐指标做两样本KS检验，并输出ECDF曲线与直方图；
    * 结构层面：计算 Spearman 相关矩阵（真实 vs 合成），输出上三角热力图和差值热力图，并打印 Frobenius 范数差。
- 绘图要求：
    * 图上所有文字均为英文（Times New Roman），但控制台输出仍为中文提示；
    * 相关矩阵热力图仅显示上三角（含对角线），并在显示的格上标注数值。
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from scipy import stats as st
from scipy.interpolate import interp1d
from scipy.special import gammaln          # t-Copula伪似然用到的Gamma函数的对数形式
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.linalg import eigh, pinv
import re

# ================= 全局绘图风格：Times New Roman =================
# 仅影响图像中文字显示；不影响控制台中文打印
plt.rcParams['font.family'] = 'Times New Roman'  # 西文字体
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 330

# ---- SciPy 1.14+ 兼容：cumtrapz 改名为 cumulative_trapezoid ----
# 部分环境中 cumtrapz 的导入路径发生变化，这里兼容处理
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except Exception:
    try:
        from scipy.integrate import cumtrapz  # 老版本名称
    except Exception:
        # 最兜底：手写一个简易版的梯形积分累计
        def cumtrapz(y, x):
            y = np.asarray(y, float); x = np.asarray(x, float)
            dx = np.diff(x)
            return np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)

# ===================== 配置区（按需修改路径和参数） =====================
EXCEL_FILE   = r"D:\Desktop\SOM-KM\数据合成\test-西山岩溶域数据-数据合成.xlsx"  # Excel路径（行=指标，列=样本 的排布）
SHEET_NAME   = 0                      # 0=第1个工作表；也可填工作表名字符串
N_SYNTH      = 232                   # 需要合成的样本数

MARGINAL     = "empirical"            # 边际分布：'empirical'（经验分布）或 'kde'（核密度）
COPULA       = "gaussian"                    # Copula：'gaussian' 或 't'
SHRINKAGE    = 0.03                   # 相关矩阵收缩（防止病态）
LOG1P_COLS   = ["NO3N","SO4","TDS","Al","Zn","Pb","Cu","Se","As","Br","Cl","F","Na","K","Ca","Mg","SiO3","HCO3","CO3"]   # 对明显右偏的变量先做 log1p 拟合边际（合成后 expm1 还原）；无需要则置空 []

# 物理约束相关：电荷平衡 + TDS近似平衡（线性投影 + 非负修复，多次迭代）
ENFORCE_CHARGE_BALANCE = True
ENFORCE_TDS_BALANCE    = True
TDS_ROW_WEIGHT         = 0.9          # TDS与各离子之和的权重系数（经验近似）
TDS_OFFSET             = -75.6        # 拟合到的截距 b
MAX_REPAIR_ITERS       = 10           # 修复迭代次数

#绘图热力图上下三角
HEATMAP_TRIANGLE = "lower"   # "upper" 上三角或 "lower"下三角


# 输出文件
OUT_XLSX      = r"D:\Desktop\SOM-KM\数据合成\synthetic_samples.xlsx"
KS_PLOTS_DIR  = r"D:\Desktop\SOM-KM\数据合成\ks_plots"
KS_REPORT_PDF = r"D:\Desktop\SOM-KM\数据合成\ks_report.pdf"
# =================================================


def chem_label(raw: str) -> str:
    r"""
    把列名（如 'HCO3(mg/L)'、'Na(mg/L)'、'NO3N'）转成 matplotlib 可渲染的上/下标写法。
    返回形如：'$\mathrm{HCO_3^{-}}$ (mg L$^{-1}$)'
    """
    s = (raw or "").strip()

    # 拆出单位（括号内），如 '(mg/L)'
    unit = ""
    m = re.search(r"\((.*?)\)\s*$", s)
    if m:
        unit = m.group(1)
        base = s[:m.start()].strip()
    else:
        base = s

    # 常见离子映射（优先精确匹配）
    mapping = {
        "Na": r"$\mathrm{Na^{+}}$",
        "K":  r"$\mathrm{K^{+}}$",
        "Ca": r"$\mathrm{Ca^{2+}}$",
        "Mg": r"$\mathrm{Mg^{2+}}$",
        "Cl": r"$\mathrm{Cl^{-}}$",
        "F":  r"$\mathrm{F^{-}}$",
        "Br": r"$\mathrm{Br^{-}}$",

        "HCO3": r"$\mathrm{HCO_3^{-}}$",
        "CO3":  r"$\mathrm{CO_3^{2-}}$",
        "SO4":  r"$\mathrm{SO_4^{2-}}$",
        "NO3":  r"$\mathrm{NO_3^{-}}$",
        "SiO3": r"$\mathrm{SiO_3^{2-}}$",

        # 以 N 计的硝酸盐（常见写法 NO3-N）
        "NO3N": r"$\mathrm{NO_3^{-}}\!-\!N$",

        # 非离子/其它（保持原样）
        "Si": r"$\mathrm{Si}$", "As":r"$\mathrm{As}$","Se":r"$\mathrm{Se}$","Pb":r"$\mathrm{Pb}$","Zn":r"$\mathrm{Zn}$",
        "Cu":r"$\mathrm{Cu}$","Al":r"$\mathrm{Al}$","TDS":"TDS","pH":"pH"
    }

    if base in mapping:
        base_tex = mapping[base]
    else:
        # 简单兜底：把化学式里的数字改为下标，例如 'SiO2' -> 'SiO_2'
        tmp = re.sub(r"([A-Za-z\)])(\d+)", r"\1_\2", base)
        base_tex = rf"$\mathrm{{{tmp}}}$"

    # 单位转写：把 'mg/L' 写成 'mg L^{-1}'（其余原样透传）
    unit_tex = ""
    if unit:
        replacements = {
            "mg/L":   r"mg/L",
            "g/L":    r"g L$^{-1}$",
            "ug/L":   r"$\mu$g L$^{-1}$",
            "μg/L":   r"$\mu$g L$^{-1}$",
            "mmol/L": r"mmol L$^{-1}$",
            "μmol/L": r"$\mu$mol L$^{-1}$",
        }
        unit_tex = replacements.get(unit, unit)
        unit_tex = f" ({unit_tex})"

    return f"{base_tex}{unit_tex}"


# ---------------- 读取 Excel（行=指标，列=样本） ----------------
def load_waterchem_excel(excel_path: str, sheet_name: Optional[object] = 0) -> pd.DataFrame:
    """
    读取Excel，并转为 DataFrame（行=样本，列=指标）。
    Excel假定结构：
      - A1 可空；
      - 第一行（B1:）是样本名；
      - 第一列（A2:）是指标名；
      - B2: 为数值区（行=指标；列=样本）。
    处理：
      - 自动裁剪到非空的最小矩形；
      - 删除全NaN的列（指标）；
      - 其他缺失用列均值填补（避免KDE/CDF报错）。
    返回：
      - df: (n_samples, n_features)
    """
    raw = pd.read_excel(excel_path, header=None, sheet_name=sheet_name)
    # 若读取到的是多表dict，取第一个
    if isinstance(raw, dict):
        if len(raw) == 0:
            raise ValueError("Excel 没有可读的工作表。")
        raw = next(iter(raw.values()))

    # 找到非空的行列并裁剪
    valid_cols = np.where(raw.notna().any(axis=0))[0]
    valid_rows = np.where(raw.notna().any(axis=1))[0]
    if len(valid_cols) == 0 or len(valid_rows) == 0:
        raise ValueError("Excel 似乎是空表。")

    raw = raw.iloc[valid_rows.min():valid_rows.max()+1, valid_cols.min():valid_cols.max()+1].copy()
    raw = raw.reset_index(drop=True)

    # 解析：样本名 & 指标名 & 数值区
    sample_names    = raw.iloc[0, 1:].astype(str).tolist()
    indicator_names = raw.iloc[1:, 0].astype(str).tolist()
    data = raw.iloc[1:, 1:].astype(float).values  # (指标×样本)

    # 转置成（样本×特征）
    X = data.T
    df = pd.DataFrame(X, columns=indicator_names, index=sample_names)
    df.index.name = "sample_id"
    df = df.reset_index(drop=False)

    # 删除全NaN的指标；其余用列均值填充
    num_cols = [c for c in df.columns if c != "sample_id"]
    all_nan = df[num_cols].isna().all(axis=0)
    if all_nan.any():
        print("[提示] 有指标全为 NaN，已剔除：", list(np.array(num_cols)[np.where(all_nan)[0]]))
        df = df.loc[:, ~all_nan.reindex(df.columns, fill_value=False).values]

    for c in df.columns:
        if c == "sample_id": continue
        col = pd.to_numeric(df[c], errors='coerce')
        mu = col.mean()
        df[c] = col.fillna(mu if np.isfinite(mu) else 0.0)

    return df.drop(columns=["sample_id"])

# ---------------- 最近 PSD 相关矩阵（确保半正定性） ----------------
def near_pd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    将任意对称矩阵“投影”为最近的正半定相关矩阵：
      1) 对称化；
      2) 特征分解，将负特征值抬到 eps；
      3) 重新合成并归一化为相关矩阵（对角为1）。
    """
    A = 0.5 * (cov + cov.T)
    vals, vecs = eigh(A)
    vals = np.maximum(vals, eps)
    A_psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(A_psd))
    A_cor = A_psd / np.outer(d, d)
    A_cor[np.diag_indices_from(A_cor)] = 1.0
    return A_cor

# ---------------- 边际模型（经验分布 / KDE） ----------------
class BaseMarginal:
    """边际分布基类：定义 fit/cdf/ppf 接口"""
    def fit(self, x: np.ndarray): ...
    def cdf(self, x: np.ndarray) -> np.ndarray: ...
    def ppf(self, u: np.ndarray) -> np.ndarray: ...

class EmpiricalMarginal(BaseMarginal):
    """
    经验分布（ECDF）：
      - 用排序样本 xs 与相应分位 us 建立 x<->u 的双向插值；
      - CDF/PPF 之间可以相互调用，不做任何参数化假设；
      - 适合样本量不大、分布形状未知的场景。
    """
    def __init__(self):
        self._ppf=None; self._cdf=None; self.xs=None; self.us=None
    def fit(self, x: np.ndarray):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 2:
            # 极端情况：只有1个点或空；构造一个近乎常数的边际，避免插值报错
            x = np.array([x[0], x[0] + 1e-9]) if x.size == 1 else np.array([0.0, 1.0])
        xs = np.sort(x); n = len(xs); us = (np.arange(1, n+1)-0.5)/n  # Blom分位
        self.xs, self.us = xs, us
        # u->x 的插值（PPF）
        self._ppf = interp1d(us, xs, bounds_error=False, assume_sorted=True,
                             fill_value=(xs[0], xs[-1]))
        # x->u 的插值（CDF）
        self._cdf = interp1d(xs, us, bounds_error=False, assume_sorted=True,
                             fill_value=(us[0], us[-1]))
        return self
    def cdf(self, x: np.ndarray) -> np.ndarray:
        u = self._cdf(x); return np.clip(u, 1e-6, 1-1e-6)  # 防止0/1导致Copula变换发散
    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1-1e-6); return self._ppf(u)

class KDEMarginal(BaseMarginal):
    """
    核密度估计边际：
      - 用 scipy.stats.gaussian_kde 得到PDF，再数值积分得到CDF；
      - **加入非负反射修正**：若样本呈非负（min>=0），则采用反射法减少边界偏差；
      - 带宽使用 0.8×Scott（略收紧，以降低过平滑）。
    """
    def __init__(self, grid_size: int=1024, bandwidth: Optional[float]=None):
        self.grid_size=grid_size; self.bandwidth=bandwidth
        self.grid_=None; self.cdf_=None; self._ppf=None; self._cdf=None; self.kde_=None
    def fit(self, x: np.ndarray):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 2:  # 极端小样本自动回退到经验分布
            return EmpiricalMarginal().fit(x)

        # 带宽：0.8 * Scott
        bw_method = (lambda s: s.scotts_factor() * 0.8) if self.bandwidth is None else self.bandwidth

        # 非负检测：若变量看起来是有 0 下界（浓度类），做反射修正
        is_nonneg = np.min(x) >= 0.0

        if is_nonneg:
            # 反射法：用原样本拟合 KDE，之后在 x>=0 上使用 f(x)+f(-x)
            kde = st.gaussian_kde(x, bw_method=bw_method)

            x_min = 0.0
            x_max = np.quantile(x, 0.999)
            pad   = 0.1*(x_max - x_min + 1e-9)
            grid  = np.linspace(x_min, x_max + pad, self.grid_size)

            # 反射后的密度（只在 x>=0 上定义）：pdf_ref(x) = f(x) + f(-x)
            pdf_pos = kde(grid) + kde(-grid)

            # 数值积分得到CDF（从 0 积到 x），并归一化
            cdf_raw = np.concatenate([[0.0], cumtrapz(pdf_pos, grid)])
            if cdf_raw[-1] > 0:
                cdf_raw /= cdf_raw[-1]

            self.grid_ = grid
            self.cdf_  = cdf_raw
            self.kde_  = kde

            # 构造 x<->u 的双向插值
            self._ppf = interp1d(self.cdf_, self.grid_, bounds_error=False, assume_sorted=True,
                                 fill_value=(self.grid_[0], self.grid_[-1]))
            self._cdf = interp1d(self.grid_, self.cdf_, bounds_error=False, assume_sorted=True,
                                 fill_value=(0.0, 1.0))
            return self

        else:
            # 一般情形（可取负）：常规 KDE
            kde = st.gaussian_kde(x, bw_method=bw_method)
            x_min, x_max = np.quantile(x, [0.001, 0.999]); pad = 0.1*(x_max-x_min+1e-9)
            grid = np.linspace(x_min-pad, x_max+pad, self.grid_size)
            pdf = kde(grid)
            # 数值积分得到CDF（并归一）
            cdf_raw = np.concatenate([[0.0], cumtrapz(pdf, grid)])
            cdf_raw /= cdf_raw[-1] if cdf_raw[-1] > 0 else 1.0
            self.grid_=grid; self.cdf_=cdf_raw; self.kde_=kde
            self._ppf = interp1d(cdf_raw, grid, bounds_error=False, assume_sorted=True,
                                 fill_value=(grid[0], grid[-1]))
            self._cdf = interp1d(grid, cdf_raw, bounds_error=False, assume_sorted=True,
                                 fill_value=(0.0, 1.0))
            return self
    def cdf(self, x: np.ndarray) -> np.ndarray:
        u = self._cdf(x); return np.clip(u, 1e-6, 1-1e-6)
    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-6, 1-1e-6); return self._ppf(u)

# ---------------- Copula（高斯 / t） ----------------
@dataclass
class GaussianCopula:
    """
    高斯Copula：
      - 对U做正态分位变换Z=Phi^{-1}(U)；
      - 用Z的相关矩阵R表征依赖结构（可做收缩与PSD修复）；
      - 采样时：先在N(0,R)中采样Z，再U=Phi(Z)。
    """
    R: Optional[np.ndarray] = None
    shrinkage: float = 0.05
    def fit(self, U: np.ndarray):
        U = np.clip(U, 1e-6, 1-1e-6); Z = st.norm.ppf(U)
        R = np.corrcoef(Z, rowvar=False)
        R = (1-self.shrinkage)*R + self.shrinkage*np.eye(R.shape[0])
        self.R = near_pd(R)  # 正半定修复
        return self
    def sample(self, n: int) -> np.ndarray:
        d = self.R.shape[0]; L = np.linalg.cholesky(self.R)
        Z = np.random.randn(n, d) @ L.T
        U = st.norm.cdf(Z)
        return np.clip(U, 1e-12, 1-1e-12)

@dataclass
class TCopula:
    """
    t-Copula：
      - 相比高斯，尾部更重，能更好地刻画极端共振；
      - 以伪似然选择最佳自由度ν，并估计相关矩阵R；
      - 采样时：Z = (L·g) / sqrt(χ²_ν/ν)，U = T_ν(Z)。
    """
    R: Optional[np.ndarray] = None
    nu: Optional[int] = None
    shrinkage: float = 0.05
    def fit(self, U: np.ndarray, nu_grid: Tuple[int,int]=(3,30)):
        U = np.clip(U, 1e-6, 1-1e-6)
        best_ll = -np.inf; best = (None, None)
        for nu in range(nu_grid[0], nu_grid[1]+1):
            Z = st.t.ppf(U, df=nu)
            R = np.corrcoef(Z, rowvar=False)
            R = (1-self.shrinkage)*R + self.shrinkage*np.eye(R.shape[0])
            R = near_pd(R)
            ll = self._pseudo_loglik_t(U, R, nu)
            if ll > best_ll:
                best_ll = ll; best = (R, nu)
        self.R, self.nu = best
        return self
    @staticmethod
    def _pseudo_loglik_t(U: np.ndarray, R: np.ndarray, nu: int) -> float:
        """
        t-Copula的伪对数似然：
          log c_T(u; R,ν) = log f_T(Z;0,R,ν) - Σ_i log f_T(z_i;0,1,ν)
          其中 Z = T_ν^{-1}(U)，f_T 为t分布密度，第一项是联合密度，第二项是各边际密度之和。
        """
        eps=1e-12; U=np.clip(U, eps, 1-eps)
        Z = st.t.ppf(U, df=nu); n, d = Z.shape
        # Cholesky 分解与二次型
        L = np.linalg.cholesky(R); Linv = np.linalg.inv(L)
        Q = np.sum((Z @ Linv.T)**2, axis=1)  # (n,)
        logdet = 2.0*np.sum(np.log(np.diag(L)))
        # 联合密度的对数（忽略常数只要相对大小即可，这里写全）
        log_mt = gammaln((nu+d)/2) - gammaln(nu/2) - 0.5*logdet - (d/2)*np.log(nu*np.pi) \
                 - ((nu+d)/2)*np.log1p(Q/nu)
        # 边际密度之和（每列标准t密度）
        log_uni = np.sum(st.t.logpdf(Z, df=nu), axis=1)
        return float(np.sum(log_mt - log_uni))
    def sample(self, n: int) -> np.ndarray:
        d = self.R.shape[0]; L = np.linalg.cholesky(self.R)
        g = np.random.randn(n, d) @ L.T
        w = np.random.chisquare(self.nu, size=(n,1))
        t = g / np.sqrt(w / self.nu)
        U = st.t.cdf(t, df=self.nu)
        return np.clip(U, 1e-12, 1-1e-12)

# ---------------- 物理约束修复（电荷 & TDS） ----------------
def _build_charge_tds_constraints(columns: list) -> tuple:
    """
    构建等式约束：
      1) 电荷平衡：Σ (z_i/M_i)*c_i = 0
      2) TDS 近似：TDS - α·Σ(离子) = b   （α=TDS_ROW_WEIGHT，b=TDS_OFFSET）
    返回：Aeq, beq, used_idx
    """
    ions = {
        "Na":{"M":22.989769,"z":+1},"K":{"M":39.0983,"z":+1},
        "Ca":{"M":40.078,"z":+2},"Mg":{"M":24.305,"z":+2},
        "Cl":{"M":35.453,"z":-1},"SO4":{"M":96.06,"z":-2},
        "HCO3":{"M":61.0168,"z":-1},"CO3":{"M":60.0089,"z":-2},
        "F":{"M":18.998403,"z":-1},"Br":{"M":79.904,"z":-1},
        "NO3":{"M":62.0049,"z":-1}, "TDS":{"M":None,"z":0},
        "Si":{"M":28.085,"z":0},"As":{"M":74.9216,"z":0},"Se":{"M":78.971,"z":0},
        "Pb":{"M":207.2,"z":0},"Zn":{"M":65.38,"z":0},"Cu":{"M":63.546,"z":0},
        "Al":{"M":26.981538,"z":0},
    }
    if "NO3N" in columns:
        ions["NO3N"] = {"M":14.0067,"z":-1}

    name_to_idx = {c:i for i,c in enumerate(columns)}
    vars_for_eq = [nm for nm in ions.keys() if nm in name_to_idx and nm!="TDS"]
    has_TDS = ("TDS" in name_to_idx)

    rows = []
    beq_rows = []   # ← 新增：每条等式的右端（电荷=0；TDS=b）

    # (1) 电荷平衡
    if ENFORCE_CHARGE_BALANCE and len(vars_for_eq)>0:
        row = np.zeros(len(columns))
        for nm in vars_for_eq:
            prop = ions[nm]; M = prop["M"]; z = prop["z"]
            if (M is not None) and z!=0:
                coef = np.sign(z)*(abs(z)/M)
                row[name_to_idx[nm]] = coef
        if np.any(row!=0):
            rows.append(row)
            beq_rows.append(0.0)  # 电荷平衡右端为 0

    # (2) TDS 近似：TDS - α·Σ(离子) = b
    if has_TDS and ENFORCE_TDS_BALANCE:
        row = np.zeros(len(columns))
        row[name_to_idx["TDS"]] = 1.0
        ion_cand = ["Na","K","Ca","Mg","Cl","SO4","HCO3","CO3","NO3","NO3N",
                    "F","Br","Si","As","Se","Pb","Zn","Cu","Al"]
        for nm in ion_cand:
            if nm in name_to_idx and nm!="TDS":
                row[name_to_idx[nm]] -= TDS_ROW_WEIGHT * 1.0  # α
        if np.any(row!=0):
            rows.append(row)
            beq_rows.append(TDS_OFFSET)  # ← 用你的截距 b

    if len(rows)==0:
        return None, None, None

    Aeq = np.vstack(rows)
    beq = np.array(beq_rows, dtype=float)   # ← 关键：不再全零，按上面逐行设置

    used_idx = np.where(np.any(Aeq!=0, axis=0))[0]
    Aeq = Aeq[:, used_idx]
    return Aeq, beq, used_idx


def _project_equalities(x: np.ndarray, Aeq: np.ndarray, beq: np.ndarray) -> np.ndarray:
    """
    将向量x投影到线性等式约束集合 {x | Aeq·x = beq}：
      x_proj = x - A^T (A A^T)^+ (A x - b)
    """
    At = Aeq.T; middle = pinv(Aeq @ At)
    return x - At @ (middle @ (Aeq @ x - beq))

def repair_physchem(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每一行样本做若干轮“线性投影 + 非负”修复：
      - 仅在 used_idx（涉及约束的变量）上修复；
      - 其他变量保持不变；
      - 每轮：投影 -> 截断为非负，重复 MAX_REPAIR_ITERS 次。
    """
    cols = list(df.columns)
    Aeq_full, beq, used_idx = _build_charge_tds_constraints(cols)
    if Aeq_full is None:
        return df.clip(lower=0)
    X = df.values.copy().astype(float)
    for i in range(X.shape[0]):
        x = X[i,:]; x_sub = x[used_idx]
        x_sub = np.maximum(x_sub, 0.0)
        for _ in range(MAX_REPAIR_ITERS):
            x_sub = _project_equalities(x_sub, Aeq_full, beq)
            x_sub = np.maximum(x_sub, 0.0)
        x[used_idx] = x_sub; X[i,:] = x
    return pd.DataFrame(X, columns=cols)

# ---------------- 合成器（边际 + Copula + 修复 + 评估） ----------------
class CopulaSynthesizer:
    """
    使用选择的边际模型（经验/KDE）与Copula（高斯/t）对整体数据进行拟合与合成。
    - fit(df):
        * 逐列拟合边际，并将原数据映射为U（0-1分位）；
        * 在U上拟合Copula的相关结构（及t自由度）。
    - sample(n):
        * 从Copula采样得到U；
        * 经各边际PPF还原为物理量纲；
        * 可选进行物理修复（电荷/TDS）。
    - full_evaluate:
        * 返回逐列KS检验结果、Spearman相关矩阵（真/合成）、以及其差的Fro范数。
    """
    def __init__(self, marginal="empirical", copula="gaussian", shrinkage=0.05,
                 log1p_cols: Optional[List[str]]=None, repair_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None):
        self.marginal=marginal; self.copula=copula; self.shrinkage=shrinkage
        self.log1p_cols=set(log1p_cols or []); self.repair_fn=repair_fn
        self.margs: Dict[str, object]={}; self.cop=None; self.columns: Optional[List[str]]=None
    def _make_marg(self):
        return EmpiricalMarginal() if self.marginal=="empirical" else KDEMarginal()
    def fit(self, df: pd.DataFrame):
        self.columns = list(df.columns)
        U = np.zeros_like(df.values, dtype=float)
        # 逐变量拟合边际，并将原值映射为U（CDF）
        for j, col in enumerate(self.columns):
            x = df[col].values.astype(float)
            if col in self.log1p_cols:  # 对显著右偏变量做对数域拟合，提升KDE稳定性
                x = np.log1p(np.maximum(x,0.0))
            marg = self._make_marg().fit(x); self.margs[col]=marg
            U[:, j] = marg.cdf(x)
        # 在U上拟合Copula结构
        if self.copula=="gaussian":
            self.cop = GaussianCopula(shrinkage=self.shrinkage).fit(U)
        elif self.copula=="t":
            self.cop = TCopula(shrinkage=self.shrinkage).fit(U)
        else:
            raise ValueError("copula must be 'gaussian' or 't'")
        return self
    def sample(self, n: int) -> pd.DataFrame:
        assert self.cop is not None and self.columns is not None
        U = self.cop.sample(n)
        Xsyn = np.zeros_like(U)
        for j, col in enumerate(self.columns):
            xj = self.margs[col].ppf(U[:, j])     # U -> 原量纲
            if col in self.log1p_cols:            # 若边际在log域拟合，这里要反变换
                xj = np.expm1(xj); xj = np.maximum(xj, 0.0)
            Xsyn[:, j] = xj
        out = pd.DataFrame(Xsyn, columns=self.columns)
        # 物理修复：电荷 & TDS
        if self.repair_fn is not None:
            out = self.repair_fn(out)
        return out
    @staticmethod
    def ks_report(real: pd.Series, synth: pd.Series) -> Tuple[float,float]:
        """
        两样本 Kolmogorov-Smirnov 检验（衡量边际分布相似度）：
          - KS_D 越小，两个分布越像；
          - KS_p 若 > 0.05，通常认为“无法拒绝分布相同”的零假设。
        """
        real = real.replace([np.inf,-np.inf], np.nan).dropna()
        synth = synth.replace([np.inf,-np.inf], np.nan).dropna()
        if len(real)<2 or len(synth)<2: return (np.nan, np.nan)
        D, p = st.ks_2samp(real.values, synth.values, alternative="two-sided", method="asymp")
        return float(D), float(p)
    @staticmethod
    def spearman_corr(df: pd.DataFrame) -> np.ndarray:
        """Spearman 等级相关矩阵（对秩做Pearson相关）"""
        return df.corr(method="spearman").values
    @staticmethod
    def corr_diff(A: np.ndarray, B: np.ndarray) -> float:
        """相关矩阵差的 Frobenius 范数（整体结构差异的一个汇总量）"""
        return float(np.linalg.norm(A-B, ord="fro"))
    def full_evaluate(self, df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict[str, object]:
        """汇总评估：KS表、Spearman矩阵（真/合成）、以及矩阵差的Fro范数"""
        ks=[]
        for col in self.columns:
            D,p = self.ks_report(df_real[col], df_syn[col]); ks.append((col,D,p))
        ks_df = pd.DataFrame(ks, columns=["variable","KS_D","KS_p"])
        R_real = self.spearman_corr(df_real[self.columns])
        R_syn  = self.spearman_corr(df_syn[self.columns])
        delta  = self.corr_diff(R_real, R_syn)
        return {"ks": ks_df, "spearman_delta_fro": delta, "R_real": R_real, "R_syn": R_syn}

# ---------------- KS/ECDF 工具函数 ----------------
def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """经验CDF曲线：返回x排序与对应累计概率"""
    xs = np.sort(x); ys = np.arange(1, len(xs)+1)/len(xs); return xs, ys

def _ks_maxdiff_point(x_real: np.ndarray, x_syn: np.ndarray) -> Tuple[float, float]:
    """计算两条ECDF之间的最大垂直差值D及其对应x位置（用于在图上标出D）"""
    xs = np.sort(np.concatenate([x_real, x_syn]))
    F_r = np.searchsorted(np.sort(x_real), xs, side='right')/len(x_real)
    F_s = np.searchsorted(np.sort(x_syn),  xs, side='right')/len(x_syn)
    diff = np.abs(F_r - F_s); i = int(np.argmax(diff))
    return float(xs[i]), float(diff[i])

def freedman_diaconis_bins(x: np.ndarray) -> int:
    """直方图分箱数（Freedman–Diaconis规则），对不同尺度较稳健"""
    x = x[np.isfinite(x)]
    if len(x)<2: return 10
    q75,q25 = np.percentile(x,[75,25]); iqr = q75-q25
    if iqr<=0: return 20
    bw = 2*iqr*(len(x)**(-1/3))
    if bw<=0: return 20
    bins = int(np.ceil((np.max(x)-np.min(x))/bw))
    return max(bins, 10)

# ---------------- 热力图工具（英文标题 + 只显示上三角/下三角） ----------------
def _annotate_matrix(ax, M: np.ndarray, fmt: str = ".2f", mask: Optional[np.ndarray] = None, cmap=None):
    """
    在热力图上标注数值：
    - 仅在 mask==True 的单元格标注（用于只标上三角/下三角）；
    - 根据底色亮度自动切换黑/白字体颜色，提升可读性。
    """
    im = ax.images[0] if ax.images else None
    if im is not None:
        norm = im.norm
        cmap = im.cmap
    else:
        norm = matplotlib.colors.Normalize(vmin=np.nanmin(M), vmax=np.nanmax(M))
        cmap = cmap or plt.get_cmap('coolwarm')

    H, W = M.shape
    if mask is None:
        mask = np.ones_like(M, dtype=bool)

    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            val = M[i, j]
            if not np.isfinite(val):
                continue
            txt = format(val, fmt)
            rgba = cmap(norm(val))
            r, g, b, _ = rgba
            # 感知亮度（近似）：决定文字为黑或白
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            color = "black" if lum > 0.55 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

def plot_corr_heatmap_upper(R: np.ndarray, labels: List[str], title: str, triangle: str = "lower"):
    """
    Spearman correlation matrix heatmap.
    triangle: 'upper' 显示上三角（含对角）；'lower' 显示下三角（含对角）
    """
    R = np.array(R, float)
    d = R.shape[0]

    # 根据参数决定显示上三角还是下三角（k=0 表示包含对角线；若想不含对角线可改为 k=1）
    if triangle.lower() == "upper":
        mask_tri = np.triu(np.ones_like(R, dtype=bool), k=0)
    elif triangle.lower() == "lower":
        mask_tri = np.tril(np.ones_like(R, dtype=bool), k=0)
    else:
        raise ValueError("triangle must be 'upper' or 'lower'")

    R_masked = np.ma.masked_where(~mask_tri, R)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad(alpha=0.0)  # 隐藏未显示的半边
    im = ax.imshow(R_masked, vmin=-1, vmax=1, interpolation="nearest", aspect="equal", cmap=cmap)

    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(d))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=11)

    # 只在显示的半边做数值标注
    _annotate_matrix(ax, R, fmt=".2f", mask=mask_tri)
    fig.tight_layout()
    return fig


def plot_corr_diff_heatmap_upper(R_syn: np.ndarray, R_real: np.ndarray, labels: List[str], title: str, triangle: str = "lower"):
    """
    Heatmap of correlation difference (Δρ = ρ_synth − ρ_real).
    triangle: 'upper' 显示上三角（含对角）；'lower' 显示下三角（含对角）
    """
    D = np.array(R_syn - R_real, float)
    d = D.shape[0]
    vmax = float(np.max(np.abs(D))); vmax = 1e-9 if vmax == 0 else vmax

    if triangle.lower() == "upper":
        mask_tri = np.triu(np.ones_like(D, dtype=bool), k=0)
    elif triangle.lower() == "lower":
        mask_tri = np.tril(np.ones_like(D, dtype=bool), k=0)
    else:
        raise ValueError("triangle must be 'upper' or 'lower'")

    D_masked = np.ma.masked_where(~mask_tri, D)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(D_masked, vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal", cmap=cmap)

    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(d))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title + f"\nFrobenius norm = {np.linalg.norm(D, ord='fro'):.3f}", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δρ = ρ_synth − ρ_real", fontsize=11)

    _annotate_matrix(ax, D, fmt=".2f", mask=mask_tri)
    fig.tight_layout()
    return fig


# ---------------- 主流程 ----------------
def main():
    os.makedirs(KS_PLOTS_DIR, exist_ok=True)

    # 1) 读取数据
    df_all = load_waterchem_excel(EXCEL_FILE, SHEET_NAME)
    print(f"[INFO] 数据规模：{df_all.shape[0]} 样本 × {df_all.shape[1]} 指标")

    # 2) 合成器：选择边际/ Copula / 对数列 / 物理修复函数
    syn = CopulaSynthesizer(
        marginal=MARGINAL, copula=COPULA, shrinkage=SHRINKAGE,
        log1p_cols=LOG1P_COLS, repair_fn=repair_physchem
    ).fit(df_all)

    # 3) 从拟合的模型采样得到“合成数据”
    X_new = syn.sample(N_SYNTH)

    # 4) 评估（KS、相关矩阵）
    rep = syn.full_evaluate(df_all, X_new)
    print("[INFO] Spearman 相关矩阵差（Fro范数）:", rep["spearman_delta_fro"])

    # ---------- 打印所有指标的 KS_D / KS_p（按 KS_D 降序） ----------
    ks_df = rep["ks"].copy().sort_values("KS_D", ascending=False).reset_index(drop=True)
    fmt = {"KS_D": "{:.6f}".format, "KS_p": "{:.6f}".format}
    print("\n[KS 全量结果（按 KS_D 降序）]")
    print(ks_df.to_string(index=False, formatters=fmt))
    valid_rows = ks_df.dropna()
    mean_D = valid_rows["KS_D"].mean()
    max_D  = valid_rows["KS_D"].max()
    frac_p = np.mean(valid_rows["KS_p"] > 0.05)
    print(f"\n[KS 汇总] 平均D={mean_D:.6f} | 最大D={max_D:.6f} | p>0.05比例={frac_p:.2%}\n")

    # 5) 输出 Excel（注意：这里将“synthetic”表转置为 行=指标、列=样本，首列为指标名）
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        # === 关键：导出为 行=指标、列=样本 ===
        X_new_T = X_new.T.copy()  # (n_features, n_samples)
        sample_cols = [f"SYN{i+1:03d}" for i in range(X_new_T.shape[1])]
        X_new_T.columns = sample_cols
        synthetic_out = X_new_T.reset_index().rename(columns={"index": "Indicator"})
        synthetic_out.to_excel(writer, sheet_name="synthetic", index=False)

        rep["ks"].to_excel(writer, sheet_name="KS", index=False)
        pd.DataFrame(rep["R_real"], columns=df_all.columns, index=df_all.columns)\
            .to_excel(writer, sheet_name="R_real")
        pd.DataFrame(rep["R_syn"],  columns=df_all.columns, index=df_all.columns)\
            .to_excel(writer, sheet_name="R_synth")
    print(f"[OK] 已写出：{OUT_XLSX}")

    # 6) 汇总 PDF：
    #    * 页1：真实相关矩阵（仅上三角，英文标题，带标注）
    #    * 页2：合成相关矩阵（同上）
    #    * 页3：差值热力图（仅上三角，附Fro范数）
    #    * 其后：逐指标ECDF+KS、直方图
    labels_raw = list(df_all.columns)
    labels_tex = [chem_label(s) for s in labels_raw]

    with PdfPages(KS_REPORT_PDF) as pdf_pages:
        # 用排版后的标签（含上下标/单位）
        fig = plot_corr_heatmap_upper(rep["R_real"], labels_tex, "Spearman Correlation (Real)", triangle=HEATMAP_TRIANGLE)
        pdf_pages.savefig(fig); plt.close(fig)

        fig = plot_corr_heatmap_upper(rep["R_syn"], labels_tex, "Spearman Correlation (Synthetic)", triangle=HEATMAP_TRIANGLE)
        pdf_pages.savefig(fig); plt.close(fig)

        fig = plot_corr_diff_heatmap_upper(rep["R_syn"], rep["R_real"], labels_tex, "Correlation Difference Heatmap", triangle=HEATMAP_TRIANGLE)
        pdf_pages.savefig(fig); plt.close(fig)

        # (4+) 逐指标 KS 图（英文标题，变量名用上下标）
        for var in df_all.columns:
            xr = df_all[var].replace([np.inf,-np.inf], np.nan).dropna().values
            xs = X_new[var].replace([np.inf,-np.inf], np.nan).dropna().values
            var_tex = chem_label(var)

            # ECDF & KS（标出最大差D）
            fig, ax = plt.subplots(figsize=(6.8,4.4))
            xr_x, xr_y = _ecdf(xr); xs_x, xs_y = _ecdf(xs)
            x_star, D = _ks_maxdiff_point(xr, xs)
            ax.step(xr_x, xr_y, where='post', label='Real ECDF')
            ax.step(xs_x, xs_y, where='post', label='Synth ECDF')
            ax.axvline(x_star, linestyle='--', color='k', alpha=0.7)
            ax.text(0.02, 0.02, f"D={D:.3f}", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"ECDF & KS: {var_tex}", fontsize=12)
            ax.set_xlabel(var_tex); ax.set_ylabel("F(x)")
            ax.legend(); ax.grid(False); plt.tight_layout()
            pdf_pages.savefig(fig); plt.close(fig)

            # 直方图
            bins = max(freedman_diaconis_bins(xr), freedman_diaconis_bins(xs))
            fig = plt.figure(figsize=(6.8,4.4))
            plt.hist(xr, bins=bins, density=True, alpha=0.5, label="Real")
            plt.hist(xs, bins=bins, density=True, alpha=0.5, label="Synth")
            plt.title(f"Histogram: {var_tex}", fontsize=12); plt.xlabel(var_tex); plt.ylabel("Density")
            plt.legend(); plt.grid(False);  plt.tight_layout()
            pdf_pages.savefig(fig); plt.close(fig)

    print(f"[OK] KS 图已汇总到：{KS_REPORT_PDF}")

# ======== 其余工具函数（便于单独调用） ========
def plot_ecdf_ks(real: pd.Series, synth: pd.Series, var: str, out_png: str, ax=None):
    """
    单变量ECDF+KS图的独立绘制函数（英文标题）：
      - 用于快速出图到PNG；
      - 主流程中已包含PDF汇总，此函数可单独调试使用。
    """
    xr = real.values; xs = synth.values
    xr = xr[np.isfinite(xr)]; xs = xs[np.isfinite(xs)]
    if ax is None:
        plt.figure(figsize=(6,4)); ax = plt.gca()
    xr_ecdf_x, xr_ecdf_y = _ecdf(xr)
    xs_ecdf_x, xs_ecdf_y = _ecdf(xs)
    x_star, D = _ks_maxdiff_point(xr, xs)
    var_tex = chem_label(var)
    ax.step(xr_ecdf_x, xr_ecdf_y, where='post', label='Real ECDF')
    ax.step(xs_ecdf_x, xs_ecdf_y, where='post', label='Synth ECDF')
    ax.axvline(x_star, linestyle='--')
    ax.text(0.02, 0.02, f"D={D:.3f}", transform=ax.transAxes)
    ax.set_title(f"ECDF & KS: {var_tex}")
    ax.set_xlabel(var_tex); ax.set_ylabel("F(x)")
    ax.legend(); ax.grid(False)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=160); plt.close()

if __name__ == "__main__":
    main()
