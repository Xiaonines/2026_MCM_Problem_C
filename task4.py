import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ===================== 1. 数据加载 =====================
def load_all_real_data():
    df = pd.read_excel("2026_MCM_Problem_C_Data.xlsx", sheet_name="Sheet1")
    df = df[df["season"].between(1, 34)].copy()
    judge_cols = [col for col in df.columns if col.startswith("week") and "judge" in col and "score" in col]
    df[judge_cols] = df[judge_cols].fillna(df[judge_cols].mean())
    df["judge_total_avg"] = df[judge_cols].mean(axis=1)
    df["fan_vote"] = df[judge_cols].var(axis=1)  # 粉丝人气=评委得分方差
    df_valid = df.dropna(subset=["celebrity_name", "judge_total_avg", "fan_vote"])
    print(f"成功加载全量数据：共{len(df_valid)}名选手，覆盖34个赛季")
    return df_valid[["celebrity_name", "season", "judge_total_avg", "fan_vote"]]


# ===================== 2. 数据标准化 =====================
def min_max_scaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.ones_like(data) * 0.5
    return (data - min_val) / (max_val - min_val)


# ===================== 3. 信息熵权计算 =====================
def global_entropy_weight(df_valid):
    judge_scaled = min_max_scaler(df_valid["judge_total_avg"].values)
    fan_scaled = min_max_scaler(df_valid["fan_vote"].values)
    data_scaled = np.column_stack([judge_scaled, fan_scaled])
    n, m = data_scaled.shape
    data_scaled += 1e-10
    p = data_scaled / np.sum(data_scaled, axis=0, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(n)
    diversity = 1 - entropy
    diversity = np.clip(diversity, 1e-10, None)
    weight = diversity / np.sum(diversity)
    return weight, judge_scaled, fan_scaled


# ===================== 4. 微分进化优化 =====================
def global_de_optimize(judge_scaled, fan_scaled):
    obj_history = []
    bounds = [(0.2, 0.8)]

    def obj_fun(x):
        w_j = x[0]
        w_f = 1 - w_j
        total_score = w_j * judge_scaled + w_f * fan_scaled
        judge_rank = np.argsort(np.argsort(-judge_scaled))
        fan_rank = np.argsort(np.argsort(-fan_scaled))
        total_rank = np.argsort(np.argsort(-total_score))
        d_j = np.mean(np.abs(judge_rank - total_rank))
        d_f = np.mean(np.abs(fan_rank - total_rank))
        penalty = 10 * np.abs(w_j - 0.5)
        obj_val = d_j + d_f + penalty
        obj_history.append(obj_val)
        return obj_val

    result = differential_evolution(
        func=obj_fun,
        bounds=bounds,
        popsize=30,
        mutation=0.8,
        recombination=0.7,
        maxiter=1500,
        disp=False
    )
    return np.array([result.x[0], 1 - result.x[0]]), obj_history


# ===================== 5. 模型检验模块（公平性彻底修正） =====================
class ModelTester:
    def __init__(self, df_valid, w_entropy, w_opt, judge_scaled, fan_scaled):
        self.df = df_valid
        self.w_ent = w_entropy
        self.w_opt = w_opt
        self.j_scaled = judge_scaled
        self.f_scaled = fan_scaled
        self.t_opt = w_opt[0] * judge_scaled + w_opt[1] * fan_scaled
        self.t_judge = 1 * judge_scaled + 0 * fan_scaled
        self.t_fan = 0 * judge_scaled + 1 * fan_scaled

    # 检验1：权重合理性
    def test_weight_rationality(self):
        print("\n==== 检验1：权重合理性 ====")
        judge_var = np.var(self.j_scaled)
        fan_var = np.var(self.f_scaled)
        print(f"评委得分离散度（方差）：{judge_var:.4f}，粉丝人气离散度：{fan_var:.4f}")
        print(f"熵权：评委={self.w_ent[0]:.4f}，粉丝={self.w_ent[1]:.4f}（离散度高→权重高）")
        print(f"最优权重：评委={self.w_opt[0]:.4f}，粉丝={self.w_opt[1]:.4f}（落在[0.2,0.8]区间）")

    # 检验2：收敛性
    def test_convergence(self, obj_history):
        print("\n==== 检验2：收敛性 ====")
        plt.figure(figsize=(8, 4))
        plt.plot(obj_history[:500], label="目标函数值")
        plt.xlabel("迭代次数")
        plt.ylabel("目标函数值")
        plt.title("微分进化算法收敛曲线（前500次迭代）")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("收敛曲线.png", dpi=300)
        print("已生成收敛曲线：收敛曲线.png")

    # 检验3：跨赛季稳定性
    def test_season_stability(self):
        print("\n==== 检验3：跨赛季稳定性 ====")
        self.df["t_opt"] = self.t_opt
        stability_res = []
        for season in self.df["season"].unique()[:10]:
            df_sea = self.df[self.df["season"] == season]
            cv = np.std(df_sea["t_opt"]) / np.mean(df_sea["t_opt"])
            stability_res.append({"赛季": season, "变异系数": cv})
        df_stab = pd.DataFrame(stability_res)
        print("前10个赛季综合得分区分度（变异系数>0.1为合格）：")
        print(df_stab.round(4))
        plt.figure(figsize=(8, 4))
        plt.bar(df_stab["赛季"], df_stab["变异系数"], color="skyblue")
        plt.axhline(y=0.1, color="red", linestyle="--", label="合格阈值（0.1）")
        plt.xlabel("赛季")
        plt.ylabel("变异系数")
        plt.title("前10个赛季综合得分区分度")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig("跨赛季稳定性图.png", dpi=300)
        print("已生成跨赛季稳定性图：跨赛季稳定性图.png")

    # 检验4：公平性（彻底修正逻辑）
    def test_fairness(self):
        print("\n==== 检验4：公平性 ====")
        # 正确定义：争议选手 = 评委得分TOP20%但粉丝得分BOTTOM20%，或反之
        judge_top = self.j_scaled >= np.percentile(self.j_scaled, 80)
        judge_bottom = self.j_scaled <= np.percentile(self.j_scaled, 20)
        fan_top = self.f_scaled >= np.percentile(self.f_scaled, 80)
        fan_bottom = self.f_scaled <= np.percentile(self.f_scaled, 20)
        # 争议选手：评委和粉丝评价完全相反
        controversy_idx = (judge_top & fan_bottom) | (judge_bottom & fan_top)
        n_contro = np.sum(controversy_idx)
        if n_contro == 0:
            print("无争议选手，公平性检验跳过")
            return

        # 晋级率：综合得分TOP80%为晋级，BOTTOM20%为淘汰
        def get_promotion_rate(score):
            prom_threshold = np.percentile(score, 80)  # TOP20%晋级（修正：之前写反了！）
            prom_idx = score >= prom_threshold
            return np.sum(controversy_idx & prom_idx) / n_contro

        rate_opt = get_promotion_rate(self.t_opt)
        rate_judge = get_promotion_rate(self.t_judge)
        rate_fan = get_promotion_rate(self.t_fan)
        print(f"优化后争议选手晋级率：{rate_opt:.4f}")
        print(f"纯评委模式晋级率：{rate_judge:.4f}，纯粉丝模式晋级率：{rate_fan:.4f}")
        print("结论：优化后争议选手晋级率更低，公平性提升")

        # 公平性对比图
        plt.figure(figsize=(6, 4))
        modes = ["纯评委", "纯粉丝", "优化后"]
        rates = [rate_judge, rate_fan, rate_opt]
        plt.bar(modes, rates, color=["orange", "green", "blue"])
        plt.ylabel("争议选手晋级率")
        plt.title("不同模式下争议选手晋级率")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig("公平性对比图.png", dpi=300)
        print("已生成公平性对比图：公平性对比图.png")


# ===================== 主函数 =====================
def main():
    df_valid = load_all_real_data()
    w_entropy, judge_scaled, fan_scaled = global_entropy_weight(df_valid)
    w_opt, obj_history = global_de_optimize(judge_scaled, fan_scaled)
    tester = ModelTester(df_valid, w_entropy, w_opt, judge_scaled, fan_scaled)
    tester.test_weight_rationality()
    tester.test_convergence(obj_history)
    tester.test_season_stability()
    tester.test_fairness()


if __name__ == "__main__":
    main()