from sklearn.tree import DecisionTreeRegressor
import torch
from tianshou.data import Batch
import warnings
import gymnasium as gym
import torch
import numpy as np
import os
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = "cuda:0"


def make_env():
    return gym.make("LunarLander-v3")


# 创建训练和测试环境
train_envs = SubprocVectorEnv([make_env for _ in range(8)])
test_envs = SubprocVectorEnv([make_env for _ in range(8)])

# 获取状态空间和动作空间的维度
state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].n
print(state_shape, action_shape)
# 创建Q网络
net = Net(state_shape, hidden_sizes=[64] * 2, action_shape=action_shape, device=DEVICE)
q_net = net.to(DEVICE)


# 创建优化器
optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)

# 创建DQN策略
policy = DQNPolicy(
    model=q_net,
    optim=optim,
    discount_factor=0.99,
    estimation_step=3,
    target_update_freq=500,
    reward_normalization=False,
    is_double=True,
    action_space=train_envs.action_space[0],
    # target_model=target_q_net
).to(DEVICE)

# Load saved model
save_path = "./"
checkpoint = torch.load(os.path.join(save_path, "dqn.pth"))
policy.load_state_dict(checkpoint["model"])
print(policy)
policy.eval()

states = np.load("./states.npy")
actions = np.load("./actions.npy")


def train_sf_dtd(policy, states, actions, max_leaf_nodes=6):
    device = next(policy.model.parameters()).device

    # 计算 Q(s,a)
    with torch.no_grad():
        batch = Batch(obs=states, info={})
        q_values = policy(batch).logits.detach().cpu().numpy()  # shape: N × action_dim

    # 取实际执行动作的 Q(s, a)
    qa = q_values[np.arange(len(actions)), actions]

    # 每一维单独训练一棵决策树
    n_features = states.shape[1]
    split_points = {}
    mses = {}

    for i in range(n_features - 2):
        xi = states[:, i].reshape(-1, 1)
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=20)
        tree.fit(xi, qa)

        # Calculate MSE
        preds = tree.predict(xi)
        mse = np.mean((preds - qa) ** 2)
        mses[i] = mse

        # 获取分裂阈值
        thresholds = tree.tree_.threshold
        usable = thresholds[thresholds != -2]  # -2 表示 leaf
        usable = np.sort(usable)

        split_points[i] = usable

    return split_points, mses


# def scan_and_plot_mse(max_leaf_nodes_list=[2, 3, 4, 5, 6, 8, 10]):
#     import matplotlib.pyplot as plt

#     # Set publication quality style (AAAI/NeurIPS style)
#     plt.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
#             "font.size": 12,
#             "axes.labelsize": 14,
#             "axes.titlesize": 14,
#             "xtick.labelsize": 12,
#             "ytick.labelsize": 12,
#             "legend.fontsize": 12,
#             "figure.figsize": (8, 4.0),
#             "lines.linewidth": 2,
#             "lines.markersize": 8,
#             "axes.grid": True,
#             "grid.alpha": 0.3,
#             "grid.linestyle": "--",
#         }
#     )

#     all_mses = {}
#     feature_names = {0: "X", 1: "Y", 2: "Vx", 3: "Vy", 4: "A", 5: "AV"}

#     print("Scanning leaf nodes for MSE plot...")
#     for nodes in max_leaf_nodes_list:
#         # print(f"Training SF-DTD with max_leaf_nodes={nodes}")
#         split_points, mses = train_sf_dtd(policy, states, actions, max_leaf_nodes=nodes)
#         for feature_idx, mse in mses.items():
#             if feature_idx not in all_mses:
#                 all_mses[feature_idx] = []
#             all_mses[feature_idx].append(mse)
#             # print(f"  Feature {feature_idx} MSE: {mse}")

#     # Plotting
#     plt.figure()

#     # Use default color cycle (usually tab10) which is good
#     for feature_idx, mses_list in all_mses.items():
#         label_name = feature_names.get(feature_idx, f"Feature {feature_idx}")
#         plt.plot(max_leaf_nodes_list, mses_list, marker="o", label=label_name)

#     plt.xlabel("Max Leaf Nodes")
#     plt.ylabel("MSE")

#     # Highlight selected size
#     plt.axvline(x=4, color="black", linestyle="--", alpha=0.6, linewidth=1.5)
#     plt.text(
#         4,
#         0.95,
#         " Selected Size",
#         transform=plt.gca().get_xaxis_transform(),
#         rotation=90,
#         verticalalignment="top",
#         color="black",
#         fontsize=12,
#     )

#     # Legend outside, top
#     plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)

#     plt.tight_layout()
#     plt.savefig("mse_plot.png", dpi=300, bbox_inches="tight")
#     plt.savefig("mse_plot.pdf", bbox_inches="tight")
#     print("Plot saved to mse_plot.png and mse_plot.pdf")


def scan_and_plot_mse(max_leaf_nodes_list=[2, 3, 4, 5, 6, 8, 10]):
    import matplotlib.pyplot as plt

    # Set publication quality style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 12,
            "mathtext.fontset": "cm",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (8, 4.0),
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    all_mses = {}
    feature_names = {
        0: "X",
        1: "Y",
        2: "$V_x$",
        3: "$V_y$",
        4: "Angle",
        5: "Angular Vel.",
    }

    print("Scanning leaf nodes for MSE plot...")
    for nodes in max_leaf_nodes_list:
        split_points, mses = train_sf_dtd(policy, states, actions, max_leaf_nodes=nodes)
        for feature_idx, mse in mses.items():
            if feature_idx not in all_mses:
                all_mses[feature_idx] = []
            all_mses[feature_idx].append(mse)

    # Plotting
    fig, ax = plt.subplots()

    for feature_idx, mses_list in all_mses.items():
        label_name = feature_names.get(feature_idx, f"Feature {feature_idx}")
        ax.plot(max_leaf_nodes_list, mses_list, marker="o", label=label_name)

    ax.set_xlabel("Number of Intervals $K$")
    ax.set_ylabel("Reconstruction Error $\mathcal{L}(K)$")
    ax.set_xticks(max_leaf_nodes_list)

    # Highlight elbow point
    ax.axvline(x=4, color="red", linestyle="--", linewidth=2, zorder=5)
    ax.annotate(
        "Elbow Point",
        xy=(4, 100),  # 箭头指向的位置
        xytext=(3, 60),  # 文字位置
        fontsize=12,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # Legend outside, top
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)

    plt.tight_layout()
    plt.savefig("mse_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig("mse_plot.pdf", bbox_inches="tight")
    print("Plot saved to mse_plot.png and mse_plot.pdf")


def generate_operators_file(split_points, filename="lunar_operators.py"):
    feature_names = {0: "X", 1: "Y", 2: "Vx", 3: "Vy", 4: "A", 5: "AV"}

    content = "lunar_operators = {\n"

    for feat_idx in range(6):
        if feat_idx not in split_points:
            continue

        splits = sorted(split_points[feat_idx])
        prefix = feature_names[feat_idx]

        if not len(splits):
            continue

        # First interval: (-inf, split[0]]
        content += (
            f'    "{prefix}1": lambda inp: (inp[:, {feat_idx}] <= {splits[0]:.5f}),\n'
        )

        # Middle intervals
        for i in range(len(splits) - 1):
            lower = splits[i]
            upper = splits[i + 1]
            content += f'    "{prefix}{i+2}": lambda inp: (inp[:, {feat_idx}] > {lower:.5f}) & (inp[:, {feat_idx}] <= {upper:.5f}),\n'

        # Last interval: (split[-1], inf)
        content += f'    "{prefix}{len(splits)+1}": lambda inp: (inp[:, {feat_idx}] > {splits[-1]:.5f}),\n'

    # Add boolean features
    content += '    "LLeg": lambda inp: inp[:, 6] == 1,\n'
    content += '    "RLeg": lambda inp: inp[:, 7] == 1,\n'
    content += "}\n"

    with open(filename, "w") as f:
        f.write(content)
    print(f"Operators saved to {filename}")


def print_splits_for_node(nodes, save_py=False):
    print(f"\n--- Split points for max_leaf_nodes={nodes} ---")
    split_points, mses = train_sf_dtd(policy, states, actions, max_leaf_nodes=nodes)
    for feature_idx, splits in split_points.items():
        print(f"Feature {feature_idx}: Split points: {splits}")
        print(f"Feature {feature_idx} MSE: {mses[feature_idx]}")

    if save_py:
        generate_operators_file(split_points)


if __name__ == "__main__":
    # 功能1：扫描leaf node num绘制折线图
    scan_and_plot_mse()

    # 功能2：在特定leaf node下输出分段
    # target_node = 4
    # print_splits_for_node(target_node, save_py=True)
