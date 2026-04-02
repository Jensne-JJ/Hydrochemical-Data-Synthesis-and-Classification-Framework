# -*- coding: utf-8 -*-
"""
功能：
1) 为每个类别分别绘制 SHAP 摘要图（蜂群图），每张包含全部特征，英文坐标与配色，保存为 PNG。
2) 允许手动输入一个或多个样本ID，在该样本所属类别上绘制单样本 SHAP 瀑布图（包含全部特征），保存为 PNG。
3) 健壮处理 shap_values 的多种返回形状：
   - (n_samples, n_features) 标准主效应
   - (n_samples, n_features, n_classes) 或 (n_samples, n_classes, n_features)：按类别拆分
   - (n_samples, n_features, n_features) 交互值：取对角线转换为主效应
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV  # ← 新增导入

import shap

import hashlib
# ========================= 基本设置 =========================
np.random.seed(1)
plt.rcParams['font.family'] = 'Times New Roman'
# 无显示环境也能保存图片
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# --------------------- 路径与参数（按需修改） ---------------------
train_excel = 'test-西山岩溶域数据.xlsx'           # Sheet1: 第1行=样本名, 第1列=特征名
label_csv   = 'som_kmeans_labels（西山数据）.csv'   # 列: sample_id, cluster
new_excel   = 'synthetic_samples(单数据).xlsx'      # sheet 'synthetic': 行=特征, 列=样本
out_pred_xlsx = 'new_predictions.xlsx'

model_choice = 'mlp'                                # 'logit' | 'svm' | 'bag' | 'knn' | 'mlp'
save_model_path = f'final_model_{model_choice}.pkl'

background_k = 50                                   # SHAP kmeans 背景点数（小=快）
output_dir = "shap_outputs"                         # 图片输出目录

# SHAP 随机数与告警控制
rng = np.random.RandomState(42)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
os.makedirs(output_dir, exist_ok=True)

# ========================= 读取训练数据 =========================
xls = pd.read_excel(train_excel, sheet_name="Sheet1", header=None)

# 样本名（第1行，从第2列开始）
train_sample_names = xls.iloc[0, 1:].values
train_sample_names = [str(x) if pd.notna(x) else f"S{idx:03d}" for idx, x in enumerate(train_sample_names)]

# 特征名（第1列，从第2行开始）
train_indicator_names = xls.iloc[1:, 0].values
train_indicator_names = [str(x) if pd.notna(x) else f"Idx{idx:03d}" for idx, x in enumerate(train_indicator_names)]

# 数值矩阵（从第2行第2列开始）
Xfull = xls.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').values

# 丢弃“整行全 NaN”的特征
valid_rows_mask = ~np.isnan(Xfull).all(axis=1)
X = Xfull[valid_rows_mask]
feature_names = [n for n, keep in zip(train_indicator_names, valid_rows_mask) if keep]

# 构建“样本为行、特征为列”的 DataFrame
X_train_df = pd.DataFrame(X.T, columns=feature_names)
X_train_df.insert(0, 'sample_id', train_sample_names)

print(f"Training matrix: {X_train_df.shape[0]} samples × {len(feature_names)} features")

# ========================= 读取标签并对齐 =========================
labels_df = pd.read_csv(label_csv)
labels_df['cluster'] = labels_df['cluster'].astype('category')
y = labels_df.set_index('sample_id').loc[train_sample_names, 'cluster'].values

X_feat = X_train_df.drop(columns='sample_id')   # (n_samples, n_features)
n_samples, n_features = X_feat.shape
# —— 仅用于绘图的漂亮特征名（不影响训练用列名/顺序）——
def make_pretty_feature_names(raw_cols):
    m = {
        'K(mg/L)'   : 'K⁺ (mg/L)',
        'Na(mg/L)'  : 'Na⁺ (mg/L)',
        'Ca(mg/L)'  : 'Ca²⁺ (mg/L)',
        'Mg(mg/L)'  : 'Mg²⁺ (mg/L)',
        'Cl(mg/L)'  : 'Cl⁻ (mg/L)',
        'CO3(mg/L)' : 'CO₃²⁻ (mg/L)',
        'HCO3(mg/L)': 'HCO₃⁻ (mg/L)',
        'SO4(mg/L)' : 'SO₄²⁻ (mg/L)',
        'F(mg/L)'   : 'F⁻ (mg/L)',
        'NO3(mg/L)' : 'NO₃⁻ (mg/L)',
        'Br(mg/L)'  : 'Br⁻ (mg/L)',
        'SiO3(mg/L)': 'SiO₃²⁻ (mg/L)',
        # 痕量元素与通用项保持元素/符号本体
        'As(mg/L)'  : 'As (mg/L)',
        'Al(mg/L)'  : 'Al (mg/L)',
        'Se(mg/L)'  : 'Se (mg/L)',
        'Cu(mg/L)'  : 'Cu (mg/L)',
        'Pb(mg/L)'  : 'Pb (mg/L)',
        'Zn(mg/L)'  : 'Zn (mg/L)',
        'TDS(mg/L)' : 'TDS (mg/L)',
        'pH'        : 'pH',
    }
    return [m.get(c, c) for c in raw_cols]

display_names = make_pretty_feature_names(X_feat.columns)


# ========================= 构建并训练模型 =========================
if model_choice == 'logit':
    estimator = LogisticRegression(max_iter=10000)
elif model_choice == 'svm':
    estimator = SVC(kernel='linear', probability=True)
elif model_choice == 'bag':
    estimator = RandomForestClassifier(n_estimators=300, random_state=1)
elif model_choice == 'knn':
    estimator = KNeighborsClassifier(n_neighbors=7)
elif model_choice == 'mlp':
    estimator = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=1000, random_state=1)
else:
    raise ValueError(f"Unknown model_choice: {model_choice}")

# Pipeline 确保标准化与特征名/顺序一致
# --------- 仅对 MLP 分支引入 CV + Grid Search；其它分支保持原逻辑不变 ----------
if model_choice == 'mlp':
    # 使用 make_pipeline 自动命名：'standardscaler' 与 'mlpclassifier'
    base_pipe = make_pipeline(StandardScaler(), estimator)

    # 网格仅搜索 MLP 的超参数；前缀使用步骤名 'mlpclassifier__'
    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(32, 16), (64, 32), (64, 32, 16)],
        'mlpclassifier__activation': ['relu', 'tanh'],
        'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
        'mlpclassifier__learning_rate_init': [0.001, 0.01],
        'mlpclassifier__solver': ['adam']  # 保持稳定性；如需也可加入 'sgd'
    }

    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=5,                  # 5 折交叉验证
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_feat, y)
    print("GridSearchCV best params:", grid_search.best_params_)
    model = grid_search.best_estimator_
else:
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_feat, y)

print(f"Model trained: {model_choice.upper()}")

# 类别信息
clf_step_name = list(model.named_steps.keys())[-1]
classes = model.named_steps[clf_step_name].classes_
class_to_idx = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)

# ========================= SHAP 工具函数（健壮处理形状） =========================
def to_main_effects_from_interactions(sv_inter: np.ndarray, n_feat: int) -> np.ndarray:
    """
    将交互 SHAP 值 (n_samples, F, F) 转为主效应 SHAP 值 (n_samples, F)，取对角线。
    """
    idx = np.arange(n_feat)
    return sv_inter[:, idx, idx]

def normalize_shap_values_shape(shap_values, model_, X_df: pd.DataFrame) -> list:
    """
    统一返回：list[ndarray]，长度=类别数（对于二分类/回归，长度=1）。
    每个 ndarray 形状为 (n_samples, n_features)，表示主效应 SHAP 值。

    自动识别以下情况：
    - list of (n_samples, n_features)  → 原样返回
    - (n_samples, n_features)          → 包装成长度1的 list
    - (n_samples, F, F)                → 交互值，取对角线 → (n_samples, F)，包装进 list
    - (n_samples, F, C) 或 (n_samples, C, F) → 按类别拆分成 list[ (n_samples, F) ]
    - 其它三维形状：若某一维为1则 squeeze，否则抛异常并提示形状
    """
    n_feat = X_df.shape[1]

    # 情况1：shap 返回 list（多输出/多类别常见）
    if isinstance(shap_values, list):
        out = []
        for arr in shap_values:
            if arr.ndim == 2 and arr.shape[1] == n_feat:
                out.append(arr)
            elif arr.ndim == 3:
                if arr.shape[1] == n_feat and arr.shape[2] == n_feat:
                    # 交互值
                    out.append(to_main_effects_from_interactions(arr, n_feat))
                elif arr.shape[1] == n_feat:
                    # (n, F, C?) -> 按最后一维拆分为 list
                    out.extend([arr[:, :, j] for j in range(arr.shape[2])])
                elif arr.shape[2] == n_feat:
                    # (n, C?, F) -> 按中间一维拆分为 list
                    out.extend([arr[:, j, :] for j in range(arr.shape[1])])
                else:
                    # 尝试 squeeze
                    if 1 in arr.shape:
                        sq = np.squeeze(arr)
                        if sq.ndim == 2 and sq.shape[1] == n_feat:
                            out.append(sq)
                        else:
                            raise ValueError(f"Unexpected 3D shape after squeeze: {sq.shape}")
                    else:
                        raise ValueError(f"Unexpected 3D shape for list element: {arr.shape}")
            else:
                raise ValueError(f"Unexpected array ndim in list element: {arr.ndim}, shape={arr.shape}")
        return out

    # 情况2：shap 返回 ndarray
    else:
        arr = shap_values
        if arr.ndim == 2 and arr.shape[1] == n_feat:
            return [arr]  # 包装为单元素 list
        elif arr.ndim == 3:
            # 交互值： (n, F, F)
            if arr.shape[1] == n_feat and arr.shape[2] == n_feat:
                return [to_main_effects_from_interactions(arr, n_feat)]
            # 类别堆叠1： (n, F, C)
            elif arr.shape[1] == n_feat:
                return [arr[:, :, j] for j in range(arr.shape[2])]
            # 类别堆叠2： (n, C, F)
            elif arr.shape[2] == n_feat:
                return [arr[:, j, :] for j in range(arr.shape[1])]
            else:
                # 尝试 squeeze
                if 1 in arr.shape:
                    sq = np.squeeze(arr)
                    if sq.ndim == 2 and sq.shape[1] == n_feat:
                        return [sq]
                    else:
                        raise ValueError(f"Unexpected 3D shape after squeeze: {sq.shape}")
                else:
                    raise ValueError(f"Unexpected 3D shape: {arr.shape}")
        else:
            raise ValueError(f"Unexpected shap_values ndim: {arr.ndim}, shape={arr.shape}")

def compute_shap_values_main_effects(model_, X_df: pd.DataFrame, background_k_: int) -> (shap.KernelExplainer, list):
    """
    计算主效应 SHAP 值，并统一返回 list[ (n_samples, n_features) ]，长度=类别数（或1）。
    """
    # 背景点数量不超过样本数且 >=1
    K = int(min(max(1, background_k_), X_df.shape[0])) if X_df.shape[0] > 0 else 1
    background = shap.kmeans(X_df, K, round_values=True)

    # 预测概率包装，保持列名
    def proba_wrapper(data_nd):
        df = pd.DataFrame(data_nd, columns=X_df.columns)
        return model_.predict_proba(df)

    explainer_ = shap.KernelExplainer(proba_wrapper, background)
    shap_values_raw = explainer_.shap_values(X_df, nsamples="auto")

    # 统一形状为 list[2D]
    shap_values_list = normalize_shap_values_shape(shap_values_raw, model_, X_df)
    return explainer_, shap_values_list

def save_current_figure(path_png: str, dpi=300):
    """保存当前图像并关闭，避免内存膨胀"""
    fig = plt.gcf()
    plt.tight_layout()
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_png}")

# ========================= 计算 SHAP（主效应，健壮形状处理） =========================
explainer, shap_values_list = compute_shap_values_main_effects(model, X_feat, background_k_=background_k)

# ========================= 1) 为每个类别绘制 SHAP 摘要图（蜂群图，全部特征） =========================
# 对于多分类，list 长度=类别数；对于二分类/回归，长度=1（视作“单一输出”）
if len(shap_values_list) == n_classes:
    # 多分类：逐类绘制
    for i, sv in enumerate(shap_values_list):
        class_name = str(classes[i])
        print(f"Drawing SHAP summary for class [{class_name}] with ALL features...")
        shap.summary_plot(
            sv,
            X_feat,
            plot_type="dot",
            max_display=n_features,  # 展示全部特征
            show=False,
            rng=rng,
            feature_names=display_names,
        )
        out_path = os.path.join(output_dir, f"shap_summary_class-{class_name}.png")
        save_current_figure(out_path)
else:
    # 非多分类（或 shap 返回仅一个输出）：只绘制一张
    print("Drawing SHAP summary (single-output) with ALL features...")
    shap.summary_plot(
        shap_values_list[0],
        X_feat,
        plot_type="dot",
        max_display=n_features,
        show=False,
        rng=rng,
        feature_names=display_names,
    )
    out_path = os.path.join(output_dir, "shap_summary.png")
    save_current_figure(out_path)

# ========================= 辅助函数：解决 Windows 文件名大小写冲突 =========================
import hashlib

def unique_wf_path(output_dir, sid, class_name):
    """
    为防止 Windows 文件系统大小写不敏感导致同名文件覆盖，
    若发现同名（仅大小写不同）文件，则自动加哈希后缀。
    """
    base = f"shap_waterfall_{sid}_class-{class_name}.png"
    path = os.path.join(output_dir, base)
    if os.name == 'nt':  # Windows系统
        existing_lower = {f.lower() for f in os.listdir(output_dir)}
        if base.lower() in existing_lower:
            h = hashlib.sha1(sid.encode('utf-8')).hexdigest()[:6]
            base = f"shap_waterfall_{sid}__{h}_class-{class_name}.png"
            path = os.path.join(output_dir, base)
    return path

# ========================= 2) 手动选择样本 → 单样本瀑布图（在该样本所属类别上绘制） =========================
sample_input = input("Enter sample IDs for single-sample SHAP waterfall (comma-separated; leave blank to skip): ").strip()
manual_sample_ids = [s.strip() for s in sample_input.split(",") if s.strip()] if sample_input else []

if manual_sample_ids:
    name_to_idx = {name: i for i, name in enumerate(train_sample_names)}

    # 该样本的“所属类别”：优先使用真实标签 y（若有），否则使用模型预测
    y_pred = model.predict(X_feat)
    true_class_idx = np.array([class_to_idx[y_i] for y_i in y], dtype=int)
    pred_class_idx = np.array([class_to_idx[c] for c in y_pred], dtype=int)

    for sid in manual_sample_ids:
        if sid not in name_to_idx:
            print(f"⚠️ Sample ID '{sid}' not found in training samples. Skipped.")
            continue

        sidx = name_to_idx[sid]

        # 选择该样本用于绘图的类别：先真后预测
        cidx = true_class_idx[sidx] if 0 <= sidx < len(true_class_idx) else pred_class_idx[sidx]
        class_name = str(classes[cidx])

        # 取该类别的 shap 行与 base value
        if len(shap_values_list) == n_classes:
            sv_row = shap_values_list[cidx][sidx]           # (n_features,)
            base_val = np.array(explainer.expected_value)[cidx]
        else:
            sv_row = shap_values_list[0][sidx]
            base_val = explainer.expected_value

        expl = shap.Explanation(
            values=sv_row,
            base_values=base_val,
            data=X_feat.iloc[sidx, :].values,
            feature_names=display_names
        )

        print(f"Drawing waterfall for sample [{sid}] on its class [{class_name}] with ALL features...")
        shap.waterfall_plot(expl, show=False, max_display=n_features)  # 强制显示全部特征
        # wf_path = os.path.join(output_dir, f"shap_waterfall_{sid}_class-{class_name}.png")
        wf_path = unique_wf_path(output_dir, sid, class_name)
        save_current_figure(wf_path)

# ========================= 保存模型（可选） =========================
# if save_model_path:
#     import joblib
#     joblib.dump(model, save_model_path)
#     print(f"Model saved: {save_model_path}")

# ========================= 预测新样本并导出 =========================
Xnew = pd.read_excel(new_excel, sheet_name="synthetic", header=0, index_col=0)
missing_in_new = [c for c in X_feat.columns if c not in Xnew.index]
extra_in_new   = [c for c in Xnew.index if c not in X_feat.columns]
if missing_in_new:
    print("NOTE: New data missing features (filled with NaN):", missing_in_new)
if extra_in_new:
    print("NOTE: New data has unseen features (ignored):", extra_in_new)

Xnew_aligned = Xnew.reindex(index=X_feat.columns)
Xnew_feat = Xnew_aligned.T

new_proba = model.predict_proba(Xnew_feat)
new_label = model.predict(Xnew_feat)
proba_df = pd.DataFrame(new_proba, columns=classes)
proba_df.insert(0, 'sample_id', Xnew_feat.index)
proba_df['pred_label'] = new_label

with pd.ExcelWriter(out_pred_xlsx) as writer:
    proba_df.to_excel(writer, sheet_name='Predictions', index=False)

print(f"\n✅ All figures saved to: {os.path.abspath(output_dir)}")
if len(shap_values_list) == n_classes:
    print("  - shap_summary_class-<CLASS>.png   (one per class, ALL features)")
else:
    print("  - shap_summary.png                 (single-output, ALL features)")
print("  - shap_waterfall_<SAMPLE>_class-<CLASS>.png (per selected sample, ALL features)")
print(f"✅ New-sample predictions saved to: {out_pred_xlsx}")
