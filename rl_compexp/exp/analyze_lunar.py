import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import formula as FM
import model as dqn_model
import settings
from data import LunarLanderDataset
from feature_extract import HookLayer
from PPO import get_PPO

GLOBALS = {}


def load_for_analysis(
    ckpt_path: str, data_path: str
) -> tuple[torch.nn.Module, Dataset]:
    model_class = getattr(dqn_model, settings.NET_NAME)
    model = model_class().to(settings.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=settings.DEVICE))
    data = np.load(data_path)
    dataset = LunarLanderDataset(data)
    return model, dataset


def load_npy(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.load(os.path.join(data_path, "states.npy"))
    acts = np.load(os.path.join(data_path, "actions.npy"))
    feats = np.load(os.path.join(data_path, "hidden_outputs.npy"))
    weights = np.load(os.path.join(data_path, "weights.npy"))
    return obs, feats, acts, weights


def extract_feature(model, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, shuffle=True, batch_size=64)
    # hook fc2 to get the feature
    hook = HookLayer()
    hook.hook_layer(layer=model.act2)
    out_list = []
    input_list = []
    feature_list = []
    # inference all data, and store the input, feature, and output
    for data in tqdm(loader):
        data = data.to(settings.DEVICE)
        with torch.no_grad():
            out = model(data)
        out_list.extend(list(out.cpu().numpy()))
        input_list.extend(list(data.cpu().numpy()))
    # list[ndarray] to list
    for feature in hook.features_blobs:
        feature_list.extend(list(feature))
    hook.features_blobs = []

    return np.array(input_list), np.array(feature_list), np.array(out_list)


def quantile_features(feats: np.ndarray) -> np.ndarray:
    # ReLU threshold
    if settings.FEATURE_THRESH is None:
        arr = feats > 0
        return np.where(arr, 1, 0).astype(np.int32)
    # threshold by FEATURE_THRESH
    else:
        quantiles = np.apply_along_axis(
            lambda a: np.quantile(a, 1 - settings.FEATURE_THRESH), 0, feats
        )
        arr = feats > quantiles[np.newaxis]
        return np.where(arr, 1, 0).astype(np.int32)


# Atomic conceptions
def gen_feat_mask(inputs) -> np.ndarray:
    ops = settings.lunarv3_operators
    feat_masks = np.zeros((len(inputs), len(ops))).astype(np.int32)
    for i, op in enumerate(ops.values()):
        feat_masks[:, i] = op(inputs)
    return feat_masks


def get_mask(feats, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # Mask has been cached
    if f.mask is not None:
        return f.mask
    if isinstance(f, FM.And):
        masks_l = get_mask(feats, f.left)
        masks_r = get_mask(feats, f.right)
        return masks_l & masks_r
    elif isinstance(f, FM.Or):
        masks_l = get_mask(feats, f.left)
        masks_r = get_mask(feats, f.right)
        return masks_l | masks_r
    elif isinstance(f, FM.Not):
        masks_val = get_mask(feats, f.val)
        return 1 - masks_val
    elif isinstance(f, FM.Leaf):
        return feats[:, f.val]
    else:
        raise ValueError("Most be passed formula")


def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)


def compute_iou(formula, acts, feats):
    masks = get_mask(feats, formula)
    # Cache mask
    formula.mask = masks
    comp_iou = iou(masks, acts)
    comp_iou = (settings.COMPLEXITY_PENALTY ** (len(formula) - 1)) * comp_iou
    return comp_iou


OPS = [(FM.Or, False), (FM.And, False), (FM.And, True)]


def compute_best_iou(args):
    (unit,) = args

    acts = GLOBALS["acts"][:, unit]
    feat_masks = GLOBALS["feat_masks"]

    if acts.sum() < settings.MIN_ACTS:
        null_f = (FM.Leaf(-1), 0)
        return {"unit": unit, "best": null_f, "best_noncomp": null_f}

    feats_to_search = list(range(feat_masks.shape[1]))
    formulas = {}
    for fval in feats_to_search:
        formula = FM.Leaf(fval)
        formulas[formula] = compute_iou(formula, acts, feat_masks)

    nonzero_iou = [k.val for k, v in formulas.items() if v > 0]
    formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))
    best_noncomp = Counter(formulas).most_common(1)[0]

    for i in range(settings.MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            # Generic binary ops
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
        # Trim the beam
        formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

    best = Counter(formulas).most_common(1)[0]

    return {
        "unit": unit,
        "best": best,
        "best_noncomp": best_noncomp,
    }


def search_feats(acts, feat_masks, feat_names, weights):
    rfile = os.path.join(settings.RESULT, f"result_{settings.MAX_FORMULA_LENGTH}.csv")
    # if os.path.exists(rfile):
    #     print(f"Loading cached {rfile}")
    #     return pd.read_csv(rfile).to_dict("records")

    GLOBALS["acts"] = acts
    GLOBALS["feat_masks"] = feat_masks

    def name(idx):
        return feat_names[idx]

    units = range(acts.shape[1])
    mp_args = [(u,) for u in units]
    pool_cls = mp.Pool
    n_done = 0
    ioufunc = compute_best_iou
    records = []
    with pool_cls(settings.PARALLEL) as pool, tqdm(
        total=len(units), desc="Units"
    ) as pbar:
        for res in pool.imap_unordered(ioufunc, mp_args):
            unit = res["unit"]
            best_lab, best_iou = res["best"]
            # w_main = weights[unit, 0]
            # w_lateral = weights[unit, 1]
            w_nothing = weights[unit, 0]
            w_left = weights[unit, 1]
            w_main = weights[unit, 2]
            w_right = weights[unit, 3]
            if best_iou > 0:
                tqdm.write(f"{unit:02d}\t{best_iou:.3f}")
            r = {
                "neuron": unit,
                "iou": round(best_iou, 3),
                "feature_length": len(best_lab),
                "best lab": best_lab.to_str(name, sort=True),
                "w_nothing": round(w_nothing, 3),
                "w_left": round(w_left, 3),
                "w_main": round(w_main, 3),
                "w_right": round(w_right, 3),
                "fire_rate": round(acts[:, unit].mean(), 3),
            }
            records.append(r)
            pbar.update()
            n_done += 1
            if n_done % settings.SAVE_EVERY == 0:
                pd.DataFrame(records).to_csv(rfile, index=False)

        # Save progress
        if len(records) % 32 == 0:
            pd.DataFrame(records).to_csv(rfile, index=False)

    pd.DataFrame(records).to_csv(rfile, index=False)
    return records


def feat2name() -> list[str]:
    names = list(settings.lunarv3_operators.keys())
    names.append("Null")
    return names


def main(
    ckpt_path="",
    data_path="/root/gym/rl_compexp/save/LunarLander-DQN64",
):
    # print("Load model and dataset")
    # model, dataset = load_for_analysis(ckpt_path, data_path)
    # weights = model.fc3.weight.t().detach().cpu().numpy()

    # print("Extract features")
    # inputs, features, outputs = extract_feature(model, dataset)
    # inputs, features, outputs, weights, _ = get_PPO(ckpt_path, data_path)
    inputs, features, outputs, weights = load_npy(data_path)
    print(features.shape)
    if not os.path.exists(settings.RESULT):
        # 如果不存在，创建路径
        os.makedirs(settings.RESULT)
    # np.save(os.path.join(settings.RESULT, f"actions.npy"), outputs)
    print("Computing quantiles")
    activations = quantile_features(features)

    print("Generating atomic masks")
    feat_masks = gen_feat_mask(inputs)

    print("search_feats")
    search_feats(activations, feat_masks, feat2name(), weights)
    res = pd.read_csv(f"{data_path}/result_5.csv")
    res = res[res["iou"] > 0]
    res = res.sort_values(by="iou", ascending=False)
    mean_iou = res["iou"].mean()
    res.to_csv(f"{data_path}/result_5_parsed.csv", index=False)


if __name__ == "__main__":
    main()
