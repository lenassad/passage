from absorb_env import AbsorbThresholdEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import networkx as nx
from utils import get_mat

# ==== 自定义回调类 ====
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        num_steps = self.num_timesteps
        if num_steps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = num_steps / elapsed
            remaining_steps = self.total_timesteps - num_steps
            est_remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else float('inf')

            if self.verbose > 0:
                print(f"[{num_steps}/{self.total_timesteps}] "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"ETA: {est_remaining_time:.1f}s")

        return True  # 返回 True 表示继续训练

# ==== 加载数据并构建环境 ====
dataset_name = 'try'
data_path = f'./data/{dataset_name}.txt'
mat0, _ = get_mat(data_path)
mat0_graph = nx.from_numpy_array(mat0, create_using=nx.Graph)

env = AbsorbThresholdEnv(mat0, mat0_graph, epsilon=1.0)

# ==== 初始化模型并训练 ====
total_steps = 100
model = PPO("MlpPolicy", env, verbose=0,n_steps=64)
progress_cb = ProgressCallback(total_timesteps=total_steps, print_freq=5)
model.learn(total_timesteps=total_steps, callback=progress_cb)

# ==== 评估训练结果 ====
obs, _ = env.reset()
action, _ = model.predict(obs)
_, reward, terminated, truncated, info = env.step(action)

print(f"reward = {reward}")
print("============ RL 搜索结果 ============")
print(f"最优吸附阈值 min_degree: {info['min_degree']}")
print(f"对应结构保真评分 reward: {info['reward']:.4f}")
print("对应各结构评价指标：")
for k in ['nmi', 'evc_overlap', 'evc_MAE', 'deg_kl', 'diam_rel', 'cc_rel', 'mod_rel']:
    print(f"{k:12s}: {info[k]:.4f}")
print("======================================")
