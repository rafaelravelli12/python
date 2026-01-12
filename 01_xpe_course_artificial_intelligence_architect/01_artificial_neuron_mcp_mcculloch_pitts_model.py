"""
    Artificial Neuron McCulloch-Pitts
    MCP Model: weighted sum of inputs + threshold-based activation function.

    :param inputs: list or tuple of inputs (binary values: 0 or 1)
    :param weights: list of weights corresponding to the inputs
    :param threshold: activation threshold
    :return: 1 (activated) or 0 (not activated)
"""

def artificial_mcp_neuron(inputs, weights, threshold):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    # z = zip(inputs, weights)
    # print(list(z))
    return 1 if weighted_sum >= threshold else 0


def simulate_and_logic():
    print("AND Logic Gate Simulation using MCP:")
    weights = [1, 1]
    threshold = 2
    possible_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for inputs in possible_inputs:
        output = artificial_mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")


def simulate_or_logic():
    print("\nOR Logic Gate Simulation using MCP:")
    weights = [1, 1]
    threshold = 1
    possible_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for inputs in possible_inputs:
        output = artificial_mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")


def simulate_not_logic():
    print("\nNOT Logic Gate Simulation using MCP:")
    weights = [-1]
    threshold = 0
    possible_inputs = [(0, ), (1, )]
    for inputs in possible_inputs:
        output = artificial_mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")


if __name__ == "__main__":
    simulate_and_logic()
    simulate_or_logic()
    simulate_not_logic()

# CODE OUTPUT:
#     AND Logic Gate Simulation using MCP:
#     Input: (0, 0) => Output: 0
#     Input: (0, 1) => Output: 0
#     Input: (1, 0) => Output: 0
#     Input: (1, 1) => Output: 1
#
#     OR Logic Gate Simulation using MCP:
#     Input: (0, 0) => Output: 0
#     Input: (0, 1) => Output: 1
#     Input: (1, 0) => Output: 1
#     Input: (1, 1) => Output: 1
#
#     NOT Logic Gate Simulation using MCP:
#     Input: (0,) => Output: 1
#     Input: (1,) => Output: 0
#
#     Process finished with exit code 0
