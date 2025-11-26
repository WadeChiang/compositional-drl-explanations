def neuron_0(state):
    x, y, vx, vy, theta, omega, l_left, l_right = state
    return (((theta > 0.3 or y < 0.4) and not (vy < -0.5)) or vy > 0.0) or (
        omega < -0.2
    )


def neuron_1(state):
    x, y, vx, vy, theta, omega, l_left, l_right = state
    return (not (theta > 0.3) and (vy <= 0.0 or vy > 0.0)) or theta < -0.3


def neuron_2(state):
    x, y, vx, vy, theta, omega, l_left, l_right = state
    return (
        (not l_left and (vx > 0.5 or vx < -0.5 or vy <= 0.0)) and not (vy > 0.0)
    ) or vy < -0.5


def neuron_3(state):
    x, y, vx, vy, theta, omega, l_left, l_right = state
    return (
        (theta >= -0.3 and theta <= 0.3 or theta > 0.3) or x >= 0.4 or x <= -0.4
    ) or l_right


def decide_actions(state):
    actions = []
    if neuron_0(state):
        actions.append("Do nothing")
    if neuron_1(state):
        actions.append("Fire left orientation engine")
    if neuron_2(state):
        actions.append("Fire main engine")
    if neuron_3(state):
        actions.append("Fire right orientation engine")
    return actions


# Example usage
state = [-0.15, 0.03, -0.13, -0.01, -0.09, -0.68, 0.00, 0.00]
actions = decide_actions(state)
print("Actions to perform:", actions)
