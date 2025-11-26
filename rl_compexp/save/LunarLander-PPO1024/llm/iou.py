import os


def read_act(path):
    with open(path, "r") as file:
        act = []
        for line in file:
            line = line.strip()
            if line == "<start>" or line == "<end>":
                continue
            stripped_line = line.strip()
            if stripped_line:
                parts = line.split()
                if len(parts) > 1:
                    number = int(parts[1])
                    act.append(number)
        return act


def read_all_acts(directory, file_pattern):
    all_acts = []
    # 列出目录中所有文件
    for filename in os.listdir(directory):
        if filename.startswith(file_pattern) and filename.endswith(".md"):
            path = os.path.join(directory, filename)
            acts = read_act(path)
            all_acts.extend(acts)
    return all_acts


correction = 0
dir = "07-17"
# 读取所有 simu 文件和 gt 文件
pred = read_all_acts(dir, "simu_")
gt = read_all_acts(dir, "gt_")
print(f"Predicted actions: {len(pred)}")
print(f"Ground truth actions: {len(gt)}")
correction += sum(1 for x, y in zip(pred, gt) if x == y)

# 计算正确率
if len(gt) > 0:
    accuracy = correction / len(gt)
else:
    accuracy = 0
print(f"Accuracy: {accuracy:.3f}")
