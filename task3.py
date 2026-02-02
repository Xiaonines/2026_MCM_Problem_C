import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. 数据加载与鲁棒性清洗
# ---------------------------------------------------------
def clean_data(file_path):
    # 解决编码问题并映射关键列
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    raw_cols = df.columns
    df['name'] = df[raw_cols[0]].astype(str)
    df['followers_raw'] = df[raw_cols[2]].astype(str)
    df['industry'] = df[raw_cols[3]].astype(str)
    df['nationality'] = df[raw_cols[5]].astype(str)
    df['age'] = pd.to_numeric(df[raw_cols[6]], errors='coerce').fillna(35)
    df['season'] = df[raw_cols[7]]
    df['placement_raw'] = df[raw_cols[9]].astype(str)

    # 正则提取数值
    df['followers'] = df['followers_raw'].str.replace(',', '').str.extract(r'(\d+)').astype(float).fillna(1)
    df['log_followers'] = np.log10(df['followers'] + 1)
    df['placement'] = df['placement_raw'].str.extract(r'(\d+)').astype(float).fillna(10)

    # 评委平均分计算
    score_cols = [c for c in raw_cols if 'judge' in str(c).lower() or 'score' in str(c).lower()]
    df['avg_score'] = df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0)

    # 粉丝支持度代理变量 (Judge Rank - Placement)
    df['judge_rank'] = df.groupby('season')['avg_score'].rank(ascending=False)
    df['fan_vote_proxy'] = df['judge_rank'] - df['placement']

    # 核心二元特征
    df['is_athlete'] = df['industry'].str.contains('Athlete', case=False, na=False).astype(int)
    df['is_us'] = df['nationality'].str.contains('United States', na=False).astype(int)
    df['is_top_3'] = (df['placement'] <= 3).astype(int)

    return df


df_model = clean_data('source.csv')

# ---------------------------------------------------------
# 2. 数学建模 (SEM 路径与多层逻辑回归)
# ---------------------------------------------------------
# SEM 路径 A: 特征 -> 评委分 (指数 GLM)
df_model['score_y'] = df_model['avg_score'] + 1
model_judge = smf.glm("score_y ~ age + log_followers + is_athlete + is_us",
                      data=df_model, family=sm.families.Gaussian(sm.families.links.Log())).fit()

# SEM 路径 B: 特征 -> 粉丝投票 (指数 GLM)
df_model['fan_y'] = df_model['fan_vote_proxy'] - df_model['fan_vote_proxy'].min() + 1
model_fan = smf.glm("fan_y ~ age + log_followers + is_athlete + is_us",
                    data=df_model, family=sm.families.Gaussian(sm.families.links.Log())).fit()

# 多层逻辑回归: 最终前三名预测
model_logit = smf.logit("is_top_3 ~ avg_score + fan_vote_proxy + C(season)", data=df_model).fit()

# ---------------------------------------------------------
# 3. 使用 matplotlib 生成图表
# ---------------------------------------------------------
influence_j = np.exp(model_judge.params).drop('Intercept')
influence_f = np.exp(model_fan.params).drop('Intercept')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图 1: 年龄 vs 评委分 (指数拟合)
axes[0, 0].scatter(df_model['age'], df_model['avg_score'], color='blue', alpha=0.5, label='Actual Data')
x_age = np.linspace(df_model['age'].min(), df_model['age'].max(), 100)
y_pred_age = np.exp(model_judge.params['Intercept'] + model_judge.params['age'] * x_age +
                    model_judge.params['log_followers'] * df_model['log_followers'].mean() +
                    model_judge.params['is_athlete'] * df_model['is_athlete'].mean() +
                    model_judge.params['is_us'] * df_model['is_us'].mean()) - 1
axes[0, 0].plot(x_age, y_pred_age, color='red', lw=3, label='Exponential Fit')
axes[0, 0].set_title('Age Impact on Judges (Exponential)', fontsize=14)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Avg Judge Score')
axes[0, 0].legend()

# 图 2: 粉丝数 vs 支持度 (指数拟合)
axes[0, 1].scatter(df_model['log_followers'], df_model['fan_vote_proxy'], color='green', alpha=0.5, label='Actual Data')
x_fol = np.linspace(df_model['log_followers'].min(), df_model['log_followers'].max(), 100)
y_pred_fol = np.exp(model_fan.params['Intercept'] + model_fan.params['age'] * df_model['age'].mean() +
                    model_fan.params['log_followers'] * x_fol +
                    model_fan.params['is_athlete'] * df_model['is_athlete'].mean() +
                    model_fan.params['is_us'] * df_model['is_us'].mean()) - (1 - df_model['fan_vote_proxy'].min())
axes[0, 1].plot(x_fol, y_pred_fol, color='orange', lw=3, label='Exponential Fit')
axes[0, 1].set_title('Follower Impact on Fan Support (Exponential)', fontsize=14)
axes[0, 1].set_xlabel('Log10(Followers)')
axes[0, 1].set_ylabel('Fan Vote Proxy')
axes[0, 1].legend()

# 图 3 & 4: 影响指数柱状图
influence_j.sort_values().plot(kind='barh', ax=axes[1, 0], color='skyblue')
axes[1, 0].axvline(1, color='black', ls='--')
axes[1, 0].set_title('Characteristic Influence Index: Judges', fontsize=14)
axes[1, 0].set_xlabel('Multiplier (exp(beta))')

influence_f.sort_values().plot(kind='barh', ax=axes[1, 1], color='salmon')
axes[1, 1].axvline(1, color='black', ls='--')
axes[1, 1].set_title('Characteristic Influence Index: Fans', fontsize=14)
axes[1, 1].set_xlabel('Multiplier (exp(beta))')

plt.tight_layout()
plt.show()