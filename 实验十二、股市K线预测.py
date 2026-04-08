import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import time


# ---------------------- 解决Matplotlib字体警告 ----------------------
def setup_stable_font():
    """直接使用Windows系统自带的SimHei字体（无需额外安装）"""
    try:
        plt.rcParams["font.family"] = "SimHei"  # Windows默认自带的中文黑体
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示
    except Exception:
        # 若SimHei也缺失，直接用英文显示避免警告
        plt.rcParams["font.family"] = "DejaVu Sans"

# 初始化字体（无警告）
setup_stable_font()
# ---------------------- 1. 数据获取（含重试+模拟） ----------------------
def get_stock_data(ticker, start_date, end_date, max_retries=3, retry_delay=2):
    for i in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            if not data.empty:
                return data['Close'].values, data.index
            print(f"第{i + 1}次重试：{ticker}数据为空")
        except Exception as e:
            print(f"第{i + 1}次重试：下载失败 - {str(e)}")
            time.sleep(retry_delay)

    print(f"⚠️  用模拟数据演示（{ticker}真实数据下载失败）")
    sim_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    sim_prices = 180 + np.cumsum(np.random.randn(len(sim_dates)) * 2)
    return sim_prices, sim_dates

# ---------------------- 2. 参数初始化 ----------------------
ticker = "AAPL"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
price_data, time_index = get_stock_data(ticker, start_date, end_date)
if len(price_data) < 2:
    raise ValueError("无有效数据，程序终止")

# ---------------------- 3. 收益率计算 ----------------------
returns = np.diff(price_data) / price_data[:-1]
returns = returns[np.abs(returns) < 3 * np.std(returns)]  # 过滤极端值

# ---------------------- 4. MH采样 ----------------------
def metropolis_hastings(returns, n_samples=10000, burn_in=2000):
    mu_init = np.mean(returns) if len(returns) else 0.0001
    sigma_init = np.std(returns) if len(returns) else 0.01
    mu, sigma = mu_init, sigma_init
    samples = []

    for _ in range(n_samples):
        # 建议分布采样
        mu_prop = norm.rvs(loc=mu, scale=0.0005)
        sigma_prop = max(norm.rvs(loc=sigma, scale=0.0005), 0.0001)

        # 对数似然+先验（避免下溢）
        log_like_curr = np.sum(norm.logpdf(returns, mu, sigma))
        log_like_prop = np.sum(norm.logpdf(returns, mu_prop, sigma_prop))
        log_prior_curr = norm.logpdf(mu, 0, 0.1) + norm.logpdf(sigma, 0, 0.1)
        log_prior_prop = norm.logpdf(mu_prop, 0, 0.1) + norm.logpdf(sigma_prop, 0, 0.1)

        # 接受概率
        log_alpha = min(0, (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr))
        if np.random.uniform() < np.exp(log_alpha):
            mu, sigma = mu_prop, sigma_prop

        samples.append((mu, sigma))

    return np.array(samples[burn_in:])

post_samples = metropolis_hastings(returns)
mu_post, sigma_post = np.mean(post_samples[:, 0]), np.mean(post_samples[:, 1])
print(f"✅ MH采样完成：μ={mu_post:.6f}, σ={sigma_post:.6f}")

# ---------------------- 5. 价格预测 ----------------------
def predict_price(last_price, mu, sigma, n_days=30):
    prices = [last_price]
    for _ in range(n_days):
        prices.append(max(prices[-1] * (1 + norm.rvs(loc=mu, scale=sigma)), 0.01))
    return np.array(prices)

pred_prices = predict_price(price_data[-1], mu_post, sigma_post)

# ---------------------- 6. 可视化 ----------------------
def plot_result(time_idx, price, pred_price, n_days=30):
    future_dates = pd.date_range(time_idx[-1] + timedelta(1), periods=n_days, freq='B')
    all_dates = time_idx.append(future_dates)

    plt.figure(figsize=(12, 6))
    plt.plot(time_idx, price, label="历史收盘价", color="#1f77b4", linewidth=1.5)
    plt.plot(all_dates[-n_days:], pred_price[1:], label=f"预测未来{n_days}天", color="#ff7f0e", linestyle="--")
    plt.scatter(time_idx[-1], price[-1], color="red", s=50, label="最后已知价格")

    plt.title(f"{ticker} 股价历史 + MH方法预测", fontsize=12)
    plt.xlabel("日期", fontsize=10)
    plt.ylabel("价格（USD）", fontsize=10)
    plt.legend(), plt.xticks(rotation=45), plt.grid(alpha=0.3)
    plt.tight_layout(), plt.show()

plot_result(time_index, price_data, pred_prices)

# ---------------------- 7. 评估 ----------------------
def evaluate(actual_returns, mu, sigma):
    if not len(actual_returns):
        print("⚠️  无数据，跳过评估")
        return
    pred_returns = norm.rvs(loc=mu, scale=sigma, size=len(actual_returns))
    mae = np.mean(np.abs(pred_returns - actual_returns))
    mse = np.mean((pred_returns - actual_returns) ** 2)
    print("\n📊 回测结果：")
    print(f"MAE：{mae:.6f}, MSE：{mse:.6f}")

evaluate(returns, mu_post, sigma_post)