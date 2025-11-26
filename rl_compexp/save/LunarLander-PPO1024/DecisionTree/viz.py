from joblib import load
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

clf = load("lunarlandertree.joblib")
feat_names = ["X", "Y", "Vx", "Vy", "Angle", "omega", "l", "r"]

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

print("决策树节点数量: ", n_nodes)

# 输出每个节点的相关信息
for i in range(n_nodes):
    if children_left[i] == children_right[i]:
        print(f"节点 {i} 是一个叶子节点，分类为 {clf.tree_.value[i]}")
    else:
        print(f"节点 {i}: 使用特征 {feature[i]} (阈值: {threshold[i]}) 进行划分")
        print(f"左子节点: {children_left[i]}, 右子节点: {children_right[i]}")
