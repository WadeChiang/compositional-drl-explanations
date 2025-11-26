import torch

DEVICE = "cuda:0"
FEATURE_THRESH = None
PARALLEL = 8
SAVE_EVERY = 4
MIN_ACTS = 500
MAX_ACTS = 7500
COMPLEXITY_PENALTY = 1.00
BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 5


lunar_operators = {
    "X1": lambda inp: (inp[:, 0] > -0.25) & (inp[:, 0] < 0),
    "X2": lambda inp: (inp[:, 0] >= 0) & (inp[:, 0] < 0.25),
    "X3": lambda inp: (inp[:, 0] > -0.4) & (inp[:, 0] <= -0.25),
    "X4": lambda inp: (inp[:, 0] < 0.4) & (inp[:, 0] >= 0.25),
    "X5": lambda inp: (inp[:, 0] <= -0.4),
    "X6": lambda inp: (inp[:, 0] >= 0.4),
    "Y1": lambda inp: (inp[:, 1] <= 0.1),
    "Y2": lambda inp: (inp[:, 1] > 0.1) & (inp[:, 1] <= 0.2),
    "Y3": lambda inp: (inp[:, 1] > 0.2) & (inp[:, 1] <= 0.3),
    "Y4": lambda inp: (inp[:, 1] > 0.3) & (inp[:, 1] <= 0.4),
    "Y5": lambda inp: (inp[:, 1] > 0.4) & (inp[:, 1] <= 0.5),
    "Y6": lambda inp: (inp[:, 1] > 0.5) & (inp[:, 1] <= 0.7),
    "Y7": lambda inp: (inp[:, 1] > 0.7),
    "Vx1": lambda inp: (inp[:, 2] >= -0.5) & (inp[:, 2] <= 0.5),  # Vx low
    "Vx2": lambda inp: (inp[:, 2] > 0.5) & (inp[:, 2] <= 1.0),  # Vx slightly positive
    "Vx3": lambda inp: (inp[:, 2] > 1.0) & (inp[:, 2] <= 2.0),  # Vx moderately positive
    "Vx4": lambda inp: (inp[:, 2] > 2.0) & (inp[:, 2] <= 5.0),  # Vx high positive
    "Vx5": lambda inp: (inp[:, 2] < -0.5) & (inp[:, 2] >= -1.0),  # Vx slightly negative
    "Vx6": lambda inp: (inp[:, 2] < -1.0)
    & (inp[:, 2] >= -2.0),  # Vx moderately negative
    "Vx7": lambda inp: (inp[:, 2] < -2.0) & (inp[:, 2] >= -5.0),
    "Vy1": lambda inp: (inp[:, 3] >= -0.5) & (inp[:, 3] <= 0.0),  # Vy low downward
    "Vy2": lambda inp: (inp[:, 3] < -0.5) & (inp[:, 3] >= -1.0),  # Vy slightly downward
    "Vy3": lambda inp: (inp[:, 3] < -1.0)
    & (inp[:, 3] >= -2.0),  # Vy moderately downward
    "Vy4": lambda inp: (inp[:, 3] < -2.0) & (inp[:, 3] >= -5.0),  # Vy high downward
    "Vy5": lambda inp: (inp[:, 3] > 0.0) & (inp[:, 3] <= 1.0),  # Vy slightly upward
    "Vy6": lambda inp: (inp[:, 3] > 1.0) & (inp[:, 3] <= 2.0),  # Vy moderately upward
    "Vy7": lambda inp: (inp[:, 3] > 2.0) & (inp[:, 3] <= 5.0),
    "A1": lambda inp: (inp[:, 4] <= -2.0),  # Angle very large (left)
    "A2": lambda inp: (inp[:, 4] > -2.0)
    & (inp[:, 4] <= -0.3),  # Angle moderately large (left)
    "A3": lambda inp: (inp[:, 4] >= -0.3) & (inp[:, 4] <= 0.3),  # Angle near 0 (center)
    "A4": lambda inp: (inp[:, 4] > 0.3)
    & (inp[:, 4] <= 2.0),  # Angle moderately large (right)
    "A5": lambda inp: (inp[:, 4] > 2.0),
    "AV1": lambda inp: (inp[:, 5] <= -3.0),  # Angular velocity very high (left)
    "AV2": lambda inp: (inp[:, 5] > -3.0)
    & (inp[:, 5] <= -0.2),  # Angular velocity moderately high (left)
    "AV3": lambda inp: (inp[:, 5] >= -0.2)
    & (inp[:, 5] <= 0.2),  # Angular velocity low (center)
    "AV4": lambda inp: (inp[:, 5] > 0.2)
    & (inp[:, 5] <= 3.0),  # Angular velocity moderately high (right)
    "AV5": lambda inp: (inp[:, 5] > 3.0),
    "LLeg": lambda inp: inp[:, 6] == 1,
    "RLeg": lambda inp: inp[:, 7] == 1,
}

lunarv3_operators = {
    "X1": lambda inp: (inp[:, 0] > -0.25) & (inp[:, 0] < 0),
    "X2": lambda inp: (inp[:, 0] >= 0) & (inp[:, 0] < 0.25),
    "X3": lambda inp: (inp[:, 0] > -0.4) & (inp[:, 0] <= -0.25),
    "X4": lambda inp: (inp[:, 0] < 0.4) & (inp[:, 0] >= 0.25),
    "X5": lambda inp: (inp[:, 0] <= -0.4),
    "X6": lambda inp: (inp[:, 0] >= 0.4),
    "Y1": lambda inp: (inp[:, 1] <= 0.1),
    "Y2": lambda inp: (inp[:, 1] > 0.1) & (inp[:, 1] <= 0.2),
    "Y3": lambda inp: (inp[:, 1] > 0.2) & (inp[:, 1] <= 0.3),
    "Y4": lambda inp: (inp[:, 1] > 0.3) & (inp[:, 1] <= 0.4),
    "Y5": lambda inp: (inp[:, 1] > 0.4) & (inp[:, 1] <= 0.5),
    "Y6": lambda inp: (inp[:, 1] > 0.5) & (inp[:, 1] <= 0.7),
    "Y7": lambda inp: (inp[:, 1] > 0.7),
    "Vx1": lambda inp: (inp[:, 2] >= -0.1) & (inp[:, 2] < 0),  # Vx low
    "Vx2": lambda inp: (inp[:, 2] >= 0) & (inp[:, 2] <= 0.1),  # Vx low
    "Vx3": lambda inp: (inp[:, 2] > 0.1) & (inp[:, 2] <= 0.2),  # Vx slightly positive
    "Vx4": lambda inp: (inp[:, 2] > 0.2) & (inp[:, 2] <= 0.4),  # Vx moderately positive
    "Vx5": lambda inp: (inp[:, 2] > 0.4) & (inp[:, 2] <= 1.0),  # Vx high positive
    "Vx6": lambda inp: (inp[:, 2] < -0.1) & (inp[:, 2] >= -0.2),  # Vx slightly negative
    "Vx7": lambda inp: (inp[:, 2] < -0.2)
    & (inp[:, 2] >= -0.4),  # Vx moderately negative
    "Vx8": lambda inp: (inp[:, 2] < -0.4) & (inp[:, 2] >= -1.0),
    "Vy1": lambda inp: (inp[:, 3] >= -0.1) & (inp[:, 3] <= 0.0),  # Vy low downward
    "Vy2": lambda inp: (inp[:, 3] < -0.1) & (inp[:, 3] >= -0.2),  # Vy slightly downward
    "Vy3": lambda inp: (inp[:, 3] < -0.2)
    & (inp[:, 3] >= -0.4),  # Vy moderately downward
    "Vy4": lambda inp: (inp[:, 3] < -0.4) & (inp[:, 3] >= -1.0),  # Vy high downward
    "Vy5": lambda inp: (inp[:, 3] > 0.0) & (inp[:, 3] <= 0.2),  # Vy slightly upward
    "Vy6": lambda inp: (inp[:, 3] > 0.2) & (inp[:, 3] <= 0.4),  # Vy moderately upward
    "Vy7": lambda inp: (inp[:, 3] > 0.4) & (inp[:, 3] <= 1.0),
    "A1": lambda inp: (inp[:, 4] <= -1.0),  # Angle very large (left)
    "A2": lambda inp: (inp[:, 4] > -1.0)
    & (inp[:, 4] <= -0.15),  # Angle moderately large (left)
    "A3": lambda inp: (inp[:, 4] >= -0.15) & (inp[:, 4] <= 0),  # Angle near 0 (center)
    "A4": lambda inp: (inp[:, 4] >= 0) & (inp[:, 4] <= 0.15),  # Angle near 0 (center)
    "A5": lambda inp: (inp[:, 4] > 0.15)
    & (inp[:, 4] <= 1.0),  # Angle moderately large (right)
    "A6": lambda inp: (inp[:, 4] > 1.0),
    "AV1": lambda inp: (inp[:, 5] <= -0.25),  # Angular velocity very high (left)
    "AV2": lambda inp: (inp[:, 5] > -0.25)
    & (inp[:, 5] <= -0.1),  # Angular velocity moderately high (left)
    "AV3": lambda inp: (inp[:, 5] >= -0.1)
    & (inp[:, 5] <= 0),  # Angular velocity low (center)
    "AV4": lambda inp: (inp[:, 5] >= 0)
    & (inp[:, 5] <= 0.1),  # Angular velocity low (center)
    "AV5": lambda inp: (inp[:, 5] > 0.1)
    & (inp[:, 5] <= 0.25),  # Angular velocity moderately high (right)
    "AV6": lambda inp: (inp[:, 5] > 0.25),
    "RLeg": lambda inp: inp[:, 6] == 1,
    "LLeg": lambda inp: inp[:, 7] == 1,
}

lunar_operators_dict = {
    "X1": "X coord in left center",
    "X2": "X coord in right center",
    "X3": "X coord in wider left center",
    "X4": "X coord in wider right center",
    "X5": "X coord outside wider left center",
    "X6": "X coord outside wider right center",
    "Y1": "Y coord very close to ground (<= 0.1)",
    "Y2": "Y coord slightly above ground (0.1, 0.2]",
    "Y3": "Y coord moderately above ground (0.2, 0.3]",
    "Y4": "Y coord near the ground upper bound (0.3, 0.4]",
    "Y5": "Y coord far from ground, lower range (0.4, 0.5]",
    "Y6": "Y coord far from ground, middle range (0.5, 0.7]",
    "Y7": "Y coord very far from ground (> 0.7)",
    "Vx1": "Vx low (between -0.5 and 0.5)",
    "Vx2": "Vx slightly positive (0.5, 1.0]",
    "Vx3": "Vx moderately positive (1.0, 2.0]",
    "Vx4": "Vx high positive (2.0, 5.0]",
    "Vx5": "Vx slightly negative (-1.0, -0.5]",
    "Vx6": "Vx moderately negative (-2.0, -1.0]",
    "Vx7": "Vx high negative (-5.0, -2.0]",
    "Vy1": "Vy low downward (-0.5 to 0.0]",
    "Vy2": "Vy slightly downward (-1.0, -0.5]",
    "Vy3": "Vy moderately downward (-2.0, -1.0]",
    "Vy4": "Vy high downward (-5.0, -2.0]",
    "Vy5": "Vy slightly upward (0.0, 1.0]",
    "Vy6": "Vy moderately upward (1.0, 2.0]",
    "Vy7": "Vy high upward (2.0, 5.0]",
    "A1": "Angle very large (left) (<= -2.0)",
    "A2": "Angle moderately large (left) (-2.0, -0.3]",
    "A3": "Angle near 0 (center) [-0.3, 0.3]",
    "A4": "Angle moderately large (right) (0.3, 2.0]",
    "A5": "Angle very large (right) (> 2.0)",
    "AV1": "Angular velocity very high (left) (<= -3.0)",
    "AV2": "Angular velocity moderately high (left) (-3.0, -0.2]",
    "AV3": "Angular velocity low (center) [-0.2, 0.2]",
    "AV4": "Angular velocity moderately high (right) (0.2, 3.0]",
    "AV5": "Angular velocity very high (right) (> 3.0)",
    "LLeg": "Left leg reaches ground",
    "RLeg": "Right leg reaches ground",
}

lunarv3_operators_dict = {
    "X1": "X coord in left center",
    "X2": "X coord in right center",
    "X3": "X coord in wider left center",
    "X4": "X coord in wider right center",
    "X5": "X coord outside wider left center",
    "X6": "X coord outside wider right center",
    "Y1": "Y coord very close to ground (<= 0.1)",
    "Y2": "Y coord slightly above ground (0.1, 0.2]",
    "Y3": "Y coord moderately above ground (0.2, 0.3]",
    "Y4": "Y coord near the ground upper bound (0.3, 0.4]",
    "Y5": "Y coord far from ground, lower range (0.4, 0.5]",
    "Y6": "Y coord far from ground, middle range (0.5, 0.7]",
    "Y7": "Y coord very far from ground (> 0.7)",
    "Vx1": "Vx low (between -0.5 and 0.5)",
    "Vx2": "Vx low (between -0.5 and 0.5)",
    "Vx3": "Vx slightly positive (0.5, 1.0]",
    "Vx4": "Vx moderately positive (1.0, 2.0]",
    "Vx5": "Vx high positive (2.0, 5.0]",
    "Vx6": "Vx slightly negative (-1.0, -0.5]",
    "Vx7": "Vx moderately negative (-2.0, -1.0]",
    "Vx8": "Vx high negative (-5.0, -2.0]",
    "Vy1": "Vy low downward (-0.5 to 0.0]",
    "Vy2": "Vy slightly downward (-1.0, -0.5]",
    "Vy3": "Vy moderately downward (-2.0, -1.0]",
    "Vy4": "Vy high downward (-5.0, -2.0]",
    "Vy5": "Vy slightly upward (0.0, 1.0]",
    "Vy6": "Vy moderately upward (1.0, 2.0]",
    "Vy7": "Vy high upward (2.0, 5.0]",
    "A1": "Angle very large (left) (<= -2.0)",
    "A2": "Angle moderately large (left) (-2.0, -0.3]",
    "A3": "Angle near 0 (center) [-0.3, 0.3]",
    "A4": "Angle moderately large (right) (0.3, 2.0]",
    "A5": "Angle very large (right) (> 2.0)",
    "AV1": "Angular velocity very high (left) (<= -3.0)",
    "AV2": "Angular velocity moderately high (left) (-3.0, -0.2]",
    "AV3": "Angular velocity low (center) [-0.2, 0.2]",
    "AV4": "Angular velocity moderately high (right) (0.2, 3.0]",
    "AV5": "Angular velocity very high (right) (> 3.0)",
    "LLeg": "Left leg reaches ground",
    "RLeg": "Right leg reaches ground",
}

def calculate_score_from_ram(byte_120, byte_121):
    """
    Calculate the score from RAM values using BCD encoding.

    Parameters:
    - byte_120 (int): The value at RAM address 120.
    - byte_121 (int): The value at RAM address 121.

    Returns:
    - int: The final score as displayed on the screen.
    """
    # Decode BCD for byte_120
    n2 = (byte_120 >> 4) & 0x0F  # High nibble
    n1 = byte_120 & 0x0F  # Low nibble

    # Decode BCD for byte_121
    n4 = (byte_121 >> 4) & 0x0F  # High nibble (十位)
    n3 = byte_121 & 0x0F  # Low nibble (个位)

    # Calculate the final score
    score = n4 * 1000 + n3 * 100 + n2 * 10 + n1

    return score


pacman_operators = {
    # 检查左侧是否有鬼靠近玩家
    "Ghost Left Close to Player": lambda inp: any(
        [
            (inp[:, 10] - inp[:, i]) < 5 and abs(inp[:, 16] - inp[:, j]) == 0
            for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
        ],
    ),
    # 检查右侧是否有鬼靠近玩家
    "Ghost Right Close to Player": lambda inp: any(
        [
            (inp[:, i] - inp[:, 10]) < 5 and abs(inp[:, 16] - inp[:, j]) == 0
            for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
        ],
    ),
    # 检查上方是否有鬼靠近玩家
    "Ghost Up Close to Player": lambda inp: any(
        [
            (inp[:, 16] - inp[:, j]) < 5 and abs(inp[:, 10] - inp[:, i]) == 0
            for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
        ],
    ),
    # 检查下方是否有鬼靠近玩家
    "Ghost Down Close to Player": lambda inp: any(
        [
            (inp[:, j] - inp[:, 16]) < 5 and abs(inp[:, 10] - inp[:, i]) == 0
            for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
        ],
    ),
    "Ghosts Far from Player": lambda inp: all(
        [
            abs(inp[:, 10] - inp[:, i]) + abs(inp[:, 16] - inp[:, j]) >= 20
            for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
        ]
    ),
    "Fruit Present": lambda inp: (inp[:, 11] != 0) & (inp[:, 17] != 0),
    "Fruit Close to Player": lambda inp: abs(inp[:, 10] - inp[:, 11])
    + abs(inp[:, 16] - inp[:, 17])
    < 20,
    "Fruit Far from Player": lambda inp: abs(inp[:, 10] - inp[:, 11])
    + abs(inp[:, 16] - inp[:, 17])
    >= 20,
    "Player Near Tunnel Entrance": lambda inp: (inp[:, 10] < 20) | (inp[:, 10] > 220),
    "Player Using Tunnel": lambda inp: (inp[:, 10] <= 5) | (inp[:, 10] >= 235),
    "High Dots Eaten": lambda inp: inp[:, 119] > 100,
    "Low Dots Eaten": lambda inp: inp[:, 119] <= 100,
    "Player Has Extra Lives": lambda inp: inp[:, 123] > 1,
    "Player On Last Life": lambda inp: inp[:, 123] == 1,
    "Player Moving Up": lambda inp: inp[:, 56] == 0,
    "Player Moving Right": lambda inp: inp[:, 56] == 1,
    "Player Moving Down": lambda inp: inp[:, 56] == 2,
    "Player Moving Left": lambda inp: inp[:, 56] == 3,
    "All Ghosts Active": lambda inp: inp[:, 19] == 4,
    "Some Ghosts Eaten": lambda inp: inp[:, 19] < 4,
    "Player Near Fruit": lambda inp: (
        abs(inp[:, 10] - inp[:, 11]) + abs(inp[:, 16] - inp[:, 17]) < 10
    ),
    "Player Far from Fruit": lambda inp: (
        abs(inp[:, 10] - inp[:, 11]) + abs(inp[:, 16] - inp[:, 17]) >= 10
    ),
    "High Risk Situation": lambda inp: (
        any(
            [
                (abs(inp[:, 10] - inp[:, i]) + abs(inp[:, 16] - inp[:, j]) < 10)
                for i, j in zip([6, 7, 8, 9], [12, 13, 14, 15])
            ]
        )
        & (inp[:, 123] == 1)  # Last life
    ),
    "Low Risk Situation": lambda inp: (
        (pacman_operators["Ghosts Far from Player"](inp)) & (inp[:, 123] > 1)
    ),
}

blackjack_operators = {f"P{i}": lambda inp, i=i: inp[:, 0] == i for i in range(1, 33)}

# 加入D1到D11算子
blackjack_operators.update(
    {f"D{i}": lambda inp, i=i: inp[:, 1] == i for i in range(1, 12)}
)

blackjack_operators.update({"HasAce": lambda inp: inp[:, 2] == 1})
blackjack_operators.update({"NoAce": lambda inp: inp[:, 2] == 0})


NET_NAME = "DQN64"
RESULT = f"../save/Blackjack2-{NET_NAME}"
