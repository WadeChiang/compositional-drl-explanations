"""
Experiment: Network Size vs Interpretability
遍历不同网络配置，统计神经元可解释性指标
"""

import os
import re
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from sklearn.tree import DecisionTreeRegressor
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from tqdm import tqdm

import formula as FM


DEVICE = "cuda:0"
BASE_DIR = "/root/gym/rl_compexp/save/LL_hyper"  
NUM_EPISODES = 5000  
MAX_SAMPLES = 10000  
MAX_LEAF_NODES = 4  


BEAM_SIZE = 50
MAX_FORMULA_LENGTH = 5
COMPLEXITY_PENALTY = 0.99
MIN_ACTS = 10
FEATURE_THRESH = None  
PARALLEL = 8

GLOBALS = {}


# ============== 1. 模型加载与解析 ==============
def parse_config(folder_name):
    """从文件夹名解析网络配置，如 LunarLander-DQN-64x3 -> (64, 3)"""
    match = re.search(r"(\d+)x(\d+)", folder_name)
    if match:
        width = int(match.group(1))
        depth = int(match.group(2))
        return width, depth
    return None, None


def load_policy(model_path, width, depth):
    """加载tianshou DQN策略"""
    env = gym.make("LunarLander-v3")
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n

    net = Net(
        state_shape,
        hidden_sizes=[width] * depth,
        action_shape=action_shape,
        device=DEVICE,
    )
    q_net = net.to(DEVICE)
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=500,
        reward_normalization=False,
        is_double=True,
        action_space=env.action_space,
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    env.close()
    return policy, state_shape, action_shape


# ============== 2. 数据采样 ==============
def collect_samples(policy, depth, num_episodes=500, max_samples=10000):
    """
    采集状态-动作样本和隐层激活
    参考你的save_hidden_outputs实现
    """
    hidden_outputs = []

    def hook_fn(module, input, output):
        hidden_outputs.append(output.detach().cpu().numpy())

    env = gym.make("LunarLander-v3")

    # policy.model.model.model 的结构:
    # depth=2时: [Linear, ReLU, Linear, ReLU, Linear]
    # depth=3时: [Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear]
    # 我们要hook最后一个隐层(倒数第二个Linear)之后的位置
    # 即索引为 2*(depth-1) 的Linear层
    last_hidden_linear_idx = 2 * (depth - 1)
    hook_handle = policy.model.model.model[
        last_hidden_linear_idx
    ].register_forward_hook(hook_fn)

    states = []
    actions = []

    for episode in range(num_episodes):
        done = False
        state, _ = env.reset()
        steps = 0

        while not done:
            states.append(state)
            batch = Batch(obs=state.reshape(1, -1), info={})
            action = policy.forward(batch).act.item()
            actions.append(action)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if len(states) >= max_samples:
                break

        if len(states) >= max_samples:
            break

    hook_handle.remove()
    env.close()

    # 截断到max_samples
    states = np.array(states[:max_samples])
    actions = np.array(actions[:max_samples])
    # hidden_outputs是list of (1, hidden_dim)，需要squeeze
    hidden_outputs = np.array([h.squeeze(0) for h in hidden_outputs[:max_samples]])

    # 获取最后一层权重 (output layer)
    # 最后一层是 policy.model.model.model[2*depth]
    output_layer_idx = 2 * depth
    weights = (
        policy.model.model.model[output_layer_idx].weight.t().detach().cpu().numpy()
    )

    return states, actions, hidden_outputs, weights


# ============== 3. 生成原子概念 ==============
def train_sf_dtd(policy, states, actions, max_leaf_nodes=4):
    """训练单特征决策树获取分割点"""
    with torch.no_grad():
        batch = Batch(obs=states, info={})
        q_values = policy(batch).logits.detach().cpu().numpy()

    qa = q_values[np.arange(len(actions)), actions]

    n_features = states.shape[1]
    split_points = {}

    for i in range(n_features - 2):  # 排除最后两个布尔特征
        xi = states[:, i].reshape(-1, 1)
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=20)
        tree.fit(xi, qa)

        thresholds = tree.tree_.threshold
        usable = thresholds[thresholds != -2]
        usable = np.sort(usable)
        split_points[i] = usable

    return split_points


def generate_operators(split_points):
    """根据分割点生成operators字典"""
    feature_names = {0: "X", 1: "Y", 2: "Vx", 3: "Vy", 4: "A", 5: "AV"}
    operators = {}

    for feat_idx in range(6):
        if feat_idx not in split_points:
            continue

        splits = sorted(split_points[feat_idx])
        prefix = feature_names[feat_idx]

        if not len(splits):
            continue

        # 第一个区间: (-inf, splits[0]]
        operators[f"{prefix}1"] = (lambda idx, s: lambda inp: inp[:, idx] <= s)(
            feat_idx, splits[0]
        )

        # 中间区间
        for i in range(len(splits) - 1):
            lower, upper = splits[i], splits[i + 1]
            operators[f"{prefix}{i+2}"] = (
                lambda idx, l, u: lambda inp: (inp[:, idx] > l) & (inp[:, idx] <= u)
            )(feat_idx, lower, upper)

        # 最后一个区间: (splits[-1], inf)
        operators[f"{prefix}{len(splits)+1}"] = (
            lambda idx, s: lambda inp: inp[:, idx] > s
        )(feat_idx, splits[-1])

    # 布尔特征
    operators["LLeg"] = lambda inp: inp[:, 6] == 1
    operators["RLeg"] = lambda inp: inp[:, 7] == 1

    return operators


# ============== 4. Beam Search 解释 ==============
def quantile_features(feats):
    """量化特征激活"""
    if FEATURE_THRESH is None:
        arr = feats > 0
        return np.where(arr, 1, 0).astype(np.int32)
    else:
        quantiles = np.apply_along_axis(
            lambda a: np.quantile(a, 1 - FEATURE_THRESH), 0, feats
        )
        arr = feats > quantiles[np.newaxis]
        return np.where(arr, 1, 0).astype(np.int32)


def gen_feat_mask(inputs, ops):
    """生成原子概念掩码"""
    feat_masks = np.zeros((len(inputs), len(ops))).astype(np.int32)
    for i, op in enumerate(ops.values()):
        feat_masks[:, i] = op(inputs)
    return feat_masks


def get_mask(feats, f):
    if f.mask is not None:
        return f.mask
    if isinstance(f, FM.And):
        return get_mask(feats, f.left) & get_mask(feats, f.right)
    elif isinstance(f, FM.Or):
        return get_mask(feats, f.left) | get_mask(feats, f.right)
    elif isinstance(f, FM.Not):
        return 1 - get_mask(feats, f.val)
    elif isinstance(f, FM.Leaf):
        return feats[:, f.val]
    else:
        raise ValueError("Must be passed formula")


def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)


def compute_iou(formula, acts, feats):
    masks = get_mask(feats, formula)
    formula.mask = masks
    comp_iou = iou(masks, acts)
    comp_iou = (COMPLEXITY_PENALTY ** (len(formula) - 1)) * comp_iou
    return comp_iou


OPS = [(FM.Or, False), (FM.And, False), (FM.And, True)]


def compute_best_iou(args):
    (unit,) = args

    acts = GLOBALS["acts"][:, unit]
    feat_masks = GLOBALS["feat_masks"]

    if acts.sum() < MIN_ACTS:
        null_f = (FM.Leaf(-1), 0)
        return {"unit": unit, "best": null_f}

    feats_to_search = list(range(feat_masks.shape[1]))
    formulas = {}
    for fval in feats_to_search:
        formula = FM.Leaf(fval)
        formulas[formula] = compute_iou(formula, acts, feat_masks)

    nonzero_iou = [k.val for k, v in formulas.items() if v > 0]
    formulas = dict(Counter(formulas).most_common(BEAM_SIZE))

    for i in range(MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            for feat in nonzero_iou:
                for op, negate in OPS:
                    if not isinstance(feat, FM.F):
                        new_formula = FM.Leaf(feat)
                    else:
                        new_formula = feat
                    if negate:
                        new_formula = FM.Not(new_formula)
                    new_formula = op(formula, new_formula)
                    new_iou = compute_iou(new_formula, acts, feat_masks)
                    new_formulas[new_formula] = new_iou

        formulas.update(new_formulas)
        formulas = dict(Counter(formulas).most_common(BEAM_SIZE))

    best = Counter(formulas).most_common(1)[0]
    return {"unit": unit, "best": best}


def search_all_neurons(activations, feat_masks, feat_names):
    """对所有神经元进行beam search"""
    GLOBALS["acts"] = activations
    GLOBALS["feat_masks"] = feat_masks

    def name(idx):
        if idx < len(feat_names):
            return feat_names[idx]
        return f"F{idx}"

    units = range(activations.shape[1])
    mp_args = [(u,) for u in units]

    records = []
    with mp.Pool(PARALLEL) as pool:
        for res in tqdm(
            pool.imap_unordered(compute_best_iou, mp_args),
            total=len(units),
            desc="Neurons",
        ):
            unit = res["unit"]
            best_lab, best_iou = res["best"]
            formula_str = best_lab.to_str(name, sort=True) if best_iou > 0 else "None"
            r = {
                "neuron": unit,
                "iou": round(best_iou, 3),
                "formula": formula_str,
            }
            records.append(r)

    return pd.DataFrame(records)


# ============== 5. 统计指标 ==============
def compute_statistics(df):
    """计算统计指标"""
    valid = df[df["iou"] > 0]

    total_neurons = len(df)
    explained_neurons = len(valid)
    unique_formulas = valid["formula"].nunique()
    mean_iou = valid["iou"].mean() if len(valid) > 0 else 0

    # 所有神经元的平均IOU（包括0）
    mean_iou_all = df["iou"].mean()

    # 高可解释神经元 (iou > 0.5)
    high_interp = len(valid[valid["iou"] > 0.5])

    return {
        "total_neurons": total_neurons,
        "explained_neurons": explained_neurons,
        "explained_ratio": (
            round(explained_neurons / total_neurons, 4) if total_neurons > 0 else 0
        ),
        "unique_formulas": unique_formulas,
        "mean_iou_valid": round(mean_iou, 4),
        "mean_iou_all": round(mean_iou_all, 4),
        "high_interp_neurons": high_interp,
        "high_interp_ratio": (
            round(high_interp / total_neurons, 4) if total_neurons > 0 else 0
        ),
    }


# ============== 6. 主流程 ==============
def process_single_config(folder_path, folder_name):
    """处理单个网络配置"""
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")

    width, depth = parse_config(folder_name)
    if width is None:
        print(f"Skipping {folder_name}: cannot parse config")
        return None

    print(f"Config: width={width}, depth={depth}")

    # 查找模型文件
    model_file = None
    for f in os.listdir(folder_path):
        if f.endswith(".pth") or f.endswith(".pt"):
            model_file = os.path.join(folder_path, f)
            break

    if model_file is None:
        print(f"No model file found in {folder_path}")
        return None

    print(f"Loading model: {model_file}")

    try:
        # 1. 加载模型
        policy, state_shape, action_shape = load_policy(model_file, width, depth)

        # 2. 采样数据
        print("Collecting samples...")
        states, actions, hidden_outputs, weights = collect_samples(
            policy, depth, NUM_EPISODES, MAX_SAMPLES
        )
        print(
            f"Collected: states={states.shape}, actions={actions.shape}, hidden={hidden_outputs.shape}, weights={weights.shape}"
        )

        # 3. 生成原子概念
        print("Generating atomic concepts...")
        split_points = train_sf_dtd(policy, states, actions, MAX_LEAF_NODES)
        operators = generate_operators(split_points)
        print(f"Generated {len(operators)} atomic concepts")

        # 4. Beam search解释
        print("Running beam search...")
        activations = quantile_features(hidden_outputs)
        feat_masks = gen_feat_mask(states, operators)
        feat_names = list(operators.keys()) + ["Null"]

        df = search_all_neurons(activations, feat_masks, feat_names)

        # 5. 统计
        stats = compute_statistics(df)
        stats["width"] = width
        stats["depth"] = depth
        stats["config"] = folder_name

        # 保存详细结果
        df.to_csv(os.path.join(folder_path, "neuron_interpretability.csv"), index=False)

        print(f"\nResults for {folder_name}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return stats

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """主函数：遍历所有配置"""
    # 扫描所有文件夹
    folders = []
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path) and "LunarLander" in item:
            folders.append((item_path, item))

    folders.sort(key=lambda x: x[1])
    print(f"Found {len(folders)} configurations to process:")
    for _, name in folders:
        print(f"  - {name}")

    # 处理每个配置
    all_results = []
    for folder_path, folder_name in folders:
        result = process_single_config(folder_path, folder_name)
        if result is not None:
            all_results.append(result)

    # 汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values(by=["width", "depth"])

        # 调整列顺序
        cols_order = [
            "config",
            "width",
            "depth",
            "total_neurons",
            "explained_neurons",
            "explained_ratio",
            "unique_formulas",
            "mean_iou_valid",
            "mean_iou_all",
            "high_interp_neurons",
            "high_interp_ratio",
        ]
        summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]

        # 保存汇总
        summary_df.to_csv(
            os.path.join(BASE_DIR, "interpretability_summary.csv"), index=False
        )

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))

        # 分析
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print("\nBy Width:")
        for width in sorted(summary_df["width"].unique()):
            subset = summary_df[summary_df["width"] == width]
            print(
                f"  Width {width}: mean_iou_valid={subset['mean_iou_valid'].mean():.4f}, explained_ratio={subset['explained_ratio'].mean():.4f}"
            )

        print("\nBy Depth:")
        for depth in sorted(summary_df["depth"].unique()):
            subset = summary_df[summary_df["depth"] == depth]
            print(
                f"  Depth {depth}: mean_iou_valid={subset['mean_iou_valid'].mean():.4f}, explained_ratio={subset['explained_ratio'].mean():.4f}"
            )


if __name__ == "__main__":
    main()
