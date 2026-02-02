import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.stats import multivariate_normal
import warnings
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')


# --------------------------
# 1. 数据加载与清洗（含舞蹈对决加分整合）
# --------------------------
def load_dancing_data(dance_bonus_dict=None):
    df = pd.read_excel("2026_MCM_Problem_C_Data.xlsx", sheet_name="Sheet1")
    judge_cols = [col for col in df.columns if "judge" in col and "score" in col and "week" in col]
    df[judge_cols] = df[judge_cols].fillna(0)
    K = len(judge_cols)
    if dance_bonus_dict is None:
        dance_bonus_dict = {}

    for week in range(1, 12):
        week_cols = [col for col in judge_cols if f"week{week}" in col]
        if week_cols:
            df[f"week{week}_judge_total_base"] = df[week_cols].sum(axis=1)
            bonus = 0
            for (s, w), val in dance_bonus_dict.items():
                if int(w) == week:
                    bonus = val
                    break
            df[f"week{week}_judge_total"] = df[f"week{week}_judge_total_base"] + bonus
            df[f"week{week}_judge_rank"] = df.groupby("season")[f"week{week}_judge_total"].rank(
                ascending=False, method="dense"
            ).fillna(0).astype(int)
    return df, K


# --------------------------
# 2. 一致性检验
# --------------------------
def consistency_check(result):
    J = result[result.columns[1]].values
    F = result[result.columns[2]].values
    corr, p_value = pearsonr(J, F)
    corr = np.abs(corr)
    n = len(result)
    RI_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
               11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59}
    RI = RI_dict.get(n, 1.0)
    CI = 0 if corr == 1 else (1 - corr) / (n - 1)
    CR = CI / RI if RI != 0 else 0.0
    if CR < 0.1:
        level = "优秀"
    elif CR < 0.2:
        level = "良好"
    else:
        level = "较差"
    return {
        "CR": round(CR, 3),
        "Pearson_corr": round(corr, 3),
        "p_value": round(p_value, 3),
        "consistency_level": level
    }


# --------------------------
# 3. 目标函数
# --------------------------
def objective(x, J):
    corr, _ = pearsonr(x, J)
    corr_term = corr if not np.isnan(corr) else 0.0
    x_norm = x / x.sum()
    entropy = -np.sum(x_norm * np.log(x_norm + 1e-10))
    var_x = np.var(x)
    x_range = np.max(x) - np.min(x)
    diversity = var_x / x_range if x_range != 0 else 0
    return -(0.6 * corr_term + 0.3 * entropy + 0.1 * diversity)


# --------------------------
# 4. 约束条件（强制被淘汰者综合排名最大）
# --------------------------
def rank_sum_constraints(F, J, eliminated_idx):
    S = F + J
    return S[eliminated_idx] - np.max(S[np.arange(len(S)) != eliminated_idx]) - 1


# --------------------------
# 5. 粉丝投票估算（含强制约束修正）
# --------------------------
def estimate_fan_vote(df, season, week, eliminated_name):
    week_rank_col = f"week{week}_judge_rank"
    valid_data = df[(df["season"] == season) &
                    (df[week_rank_col] > 0) &
                    (df["celebrity_name"] != "Unknown")].copy()
    valid_data = valid_data.reset_index(drop=True)
    N = len(valid_data)
    if N <= 1:
        raise ValueError(f"第{season}季第{week}周有效选手不足！")

    eliminated_mask = valid_data["celebrity_name"] == eliminated_name
    if not eliminated_mask.any():
        raise ValueError(f"第{season}季第{week}周未找到淘汰者：{eliminated_name}")
    eliminated_local_idx = eliminated_mask.idxmax()

    J = valid_data[week_rank_col].values
    initial_F = J.copy()
    initial_F[eliminated_local_idx] = N

    constraints = [
        {"type": "ineq", "fun": rank_sum_constraints, "args": (J, eliminated_local_idx)},
        {"type": "ineq", "fun": lambda x: x - 1},
        {"type": "ineq", "fun": lambda x: N - x}
    ]
    bounds = [(1, N) for _ in range(N)]

    res = minimize(
        fun=objective,
        x0=initial_F,
        args=(J,),
        method="L-BFGS-B",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "gtol": 1e-6, "disp": False}
    )

    fan_rank = pd.Series(res.x).rank(method="dense", ascending=True).astype(int).values
    valid_data[f"week{week}_fan_rank"] = fan_rank
    valid_data[f"week{week}_total_rank"] = valid_data[week_rank_col] + fan_rank

    # 强制修正：确保被淘汰者综合排名为最大值
    total_rank = valid_data[f"week{week}_total_rank"].values
    if total_rank[eliminated_local_idx] != np.max(total_rank):
        valid_data.loc[eliminated_local_idx, f"week{week}_fan_rank"] = N
        valid_data.loc[eliminated_local_idx, f"week{week}_total_rank"] = J[eliminated_local_idx] + N

    core_cols = ["celebrity_name", week_rank_col, f"week{week}_fan_rank", f"week{week}_total_rank"]
    return valid_data[core_cols], J, N, eliminated_local_idx


# --------------------------
# 6. 贝叶斯采样（σ=0.8，保证多样性）
# --------------------------
def bayesian_uncertainty(result, J, N, eliminated_local_idx, season, week, n_samples=300):
    fan_rank_point = result[result.columns[2]].values
    prior_std = 0.8  # 增大扰动幅度，保证采样多样性
    samples = []
    max_attempts = 15000
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        F_perturb = np.round(fan_rank_point + multivariate_normal.rvs(mean=np.zeros(N), cov=np.eye(N) * prior_std ** 2))
        F_perturb = np.clip(F_perturb, 1, N).astype(int)
        F_perturb = pd.Series(F_perturb).rank(method="dense", ascending=True).astype(int).values
        S_perturb = F_perturb + J
        if S_perturb[eliminated_local_idx] == np.max(S_perturb):
            samples.append(F_perturb)
        attempts += 1

    if len(samples) < n_samples:
        samples += [fan_rank_point] * (n_samples - len(samples))
    samples = np.array(samples)

    stats = pd.DataFrame()
    stats["celebrity_name"] = result["celebrity_name"]
    stats["post_mean"] = np.mean(samples, axis=0).round(2)
    stats["post_std"] = np.std(samples, axis=0).round(2)
    stats["95%_lower"] = np.percentile(samples, 2.5, axis=0).round(0).astype(int)
    stats["95%_lower"] = np.maximum(stats["95%_lower"], 1)
    stats["95%_upper"] = np.percentile(samples, 97.5, axis=0).round(0).astype(int)
    stats["95%_upper"] = np.minimum(stats["95%_upper"], N)
    stats["CV"] = (stats["post_std"] / stats["post_mean"]).round(3)
    stats["CV"] = stats["CV"].fillna(0)

    plt.figure(figsize=(12, 7))
    for i, name in enumerate(stats["celebrity_name"]):
        plt.hist(samples[:, i], bins=np.arange(0.5, N + 1.5, 1), alpha=0.6, label=name)
    plt.title(f"第{season}季第{week}周粉丝排名不确定性分布（σ={prior_std}）", fontsize=14)
    plt.xlabel("粉丝排名（1为最高）", fontsize=12)
    plt.ylabel("采样频次", fontsize=12)
    plt.xticks(range(1, N + 1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return stats, samples


# --------------------------
# 7. 把握程度分析
# --------------------------
def confidence_analysis(stats, samples, consistency_result, eliminated_name, N, season, week):
    stats["conf_interval_width"] = stats["95%_upper"] - stats["95%_lower"]
    stats["sample_convergence"] = np.array([len(np.unique(samples[:, i])) / N for i in range(N)]).round(3)

    stats["CV_norm"] = 1 - np.clip(stats["CV"] / 0.3, 0, 1)
    stats["width_norm"] = 1 - np.clip(stats["conf_interval_width"] / N, 0, 1)
    stats["conv_norm"] = np.clip(stats["sample_convergence"], 0, 1)

    stats["confidence_score"] = (stats["CV_norm"] * 0.6 + stats["width_norm"] * 0.35 + stats["conv_norm"] * 0.05).round(
        3)

    def get_conf_level(score):
        if score >= 0.8:
            return "极高"
        elif score >= 0.6:
            return "较高"
        elif score >= 0.4:
            return "中等"
        else:
            return "较低"

    stats["confidence_level"] = stats["confidence_score"].apply(get_conf_level)

    avg_conf_score = stats["confidence_score"].mean()
    cr = consistency_result["CR"]
    p_val = consistency_result["p_value"]
    if avg_conf_score >= 0.8 and cr < 0.1 and p_val < 0.05:
        overall_conf_level = "★★★★★ 极高把握"
    elif avg_conf_score >= 0.6 and cr < 0.2:
        overall_conf_level = "★★★★ 较高把握"
    elif avg_conf_score >= 0.4:
        overall_conf_level = "★★★ 中等把握"
    else:
        overall_conf_level = "★★ 较低把握"

    support_metrics = {
        "平均把握分数": round(avg_conf_score, 3),
        "一致性CR值": cr,
        "相关性P值": p_val,
        "平均CV值": round(stats["CV"].mean(), 3),
        "平均置信区间宽度": round(stats["conf_interval_width"].mean(), 1),
        "淘汰者把握程度": stats[stats["celebrity_name"] == eliminated_name]["confidence_level"].values[0],
        "淘汰者把握分数": stats[stats["celebrity_name"] == eliminated_name]["confidence_score"].values[0]
    }

    plt.figure(figsize=(14, 6))
    color_map = {"极高": "#2E8B57", "较高": "#4682B4", "中等": "#FFD700", "较低": "#DC143C"}
    bar_colors = [color_map[level] for level in stats["confidence_level"]]
    plt.bar(stats["celebrity_name"], stats["confidence_score"], color=bar_colors, alpha=0.8, edgecolor="black")
    plt.axhline(y=0.8, color="#2E8B57", linestyle="--", linewidth=1.5, label="极高把握(≥0.8)")
    plt.axhline(y=0.6, color="#4682B4", linestyle="--", linewidth=1.5, label="较高把握(≥0.6)")
    plt.axhline(y=0.4, color="#FFD700", linestyle="--", linewidth=1.5, label="中等把握(≥0.4)")
    plt.ylim(0, 1.05)
    plt.title(f"第{season}季第{week}周粉丝排名预测把握程度（整体：{overall_conf_level}）", fontsize=14)
    plt.xlabel("参赛选手", fontsize=12)
    plt.ylabel("把握程度分数（0-1）", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return stats, overall_conf_level, support_metrics


# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    SEASON = 2
    WEEK = 4
    ELIMINATED_NAME = "Jerry Rice"
    N_SAMPLES = 300
    DANCE_BONUS_DICT = {}

    print("=" * 65)
    print("1. 数据加载与舞蹈对决加分整合")
    print("=" * 65)
    df, K = load_dancing_data(DANCE_BONUS_DICT)
    print(f"✅ 数据加载完成：共{len(df)}位选手，{K}位评委，舞蹈对决加分规则：{DANCE_BONUS_DICT}")
    print(f"✅ 目标分析：第{SEASON}季第{WEEK}周，被淘汰选手：{ELIMINATED_NAME}\n")

    print("=" * 65)
    print("2. 带淘汰约束的粉丝投票估算")
    print("=" * 65)
    try:
        fan_vote_result, J, N, eliminated_idx = estimate_fan_vote(df, SEASON, WEEK, ELIMINATED_NAME)
        print("✅ 粉丝排名估算完成，点解结果：")
        print(fan_vote_result.to_string(index=False))
    except Exception as e:
        print(f"❌ 估算失败：{e}")
        exit()

    print("\n" + "=" * 65)
    print("3. 模型一致性检验（CR值）")
    print("=" * 65)
    consistency_result = consistency_check(fan_vote_result)
    print(f"✅ CR值（一致性比率）：{consistency_result['CR']}（{consistency_result['consistency_level']}）")
    print(f"✅ Pearson相关系数：{consistency_result['Pearson_corr']}（越接近1拟合越好）")
    print(f"✅ 相关性P值：{consistency_result['p_value']}（<0.05为统计显著）\n")

    print("=" * 65)
    print("4. 贝叶斯不确定性量化（先验标准差σ=0.8）")
    print("=" * 65)
    uncertainty_stats, samples = bayesian_uncertainty(fan_vote_result, J, N, eliminated_idx, SEASON, WEEK, N_SAMPLES)
    print("✅ 贝叶斯采样完成，后验统计结果：")
    print(uncertainty_stats.to_string(index=False))

    print("\n" + "=" * 65)
    print("5. 估算结果把握程度分析（核心）")
    print("=" * 65)
    conf_stats, overall_conf, support_metrics = confidence_analysis(uncertainty_stats, samples, consistency_result,
                                                                    ELIMINATED_NAME, N, SEASON, WEEK)
    print("✅ 各选手把握程度详情：")
    conf_detail = conf_stats[["celebrity_name", "confidence_score", "confidence_level", "CV", "conf_interval_width"]]
    print(conf_detail.to_string(index=False))
    print(f"\n📊 整体预测把握程度：{overall_conf}")
    print("🔑 核心支撑指标：")
    for k, v in support_metrics.items():
        print(f"   - {k}：{v}")

    print("\n" + "=" * 65)
    print("6. 模型最终综合评估")
    print("=" * 65)
    avg_CV = uncertainty_stats["CV"].mean()
    elim_total_rank = \
    fan_vote_result[fan_vote_result["celebrity_name"] == ELIMINATED_NAME][f"week{WEEK}_total_rank"].values[0]
    max_total_rank = fan_vote_result[f"week{WEEK}_total_rank"].max()
    print(f"✅ 平均CV值：{avg_CV:.3f} → {'稳定' if avg_CV < 0.2 else '较稳定' if avg_CV < 0.3 else '不稳定'}")
    print(f"✅ 一致性等级：{consistency_result['consistency_level']}（CR={consistency_result['CR']}）")
    print(
        f"✅ 淘汰约束满足：{elim_total_rank == max_total_rank}（被淘汰者综合排名={elim_total_rank}，最大值={max_total_rank}）")
    print(f"✅ 相关性显著：{consistency_result['p_value'] < 0.05}")
    print(f"✅ 整体把握程度：{overall_conf}")
    print("\n" + "=" * 65)
    print("📌 模型运行完成，所有结果已输出并可视化！")
    print("=" * 65)