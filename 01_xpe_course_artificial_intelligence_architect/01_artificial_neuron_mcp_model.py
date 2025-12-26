"""
    Artificial Neuron McCulloch-Pitts
    MCP Model: weighted sum of inputs + threshold-based activation function.

    :param inputs: list or tuple of inputs (binary values: 0 or 1)
    :param weights: list of weights corresponding to the inputs
    :param threshold: activation threshold
    :return: 1 (activated) or 0 (not activated)
"""

def mcp_neuron(inputs, weights, threshold):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0

def simulate_and():
    print("AND Logic Gate Simulation using MCP:")
    weights = [1, 1]
    threshold = 2
    possible_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for inputs in possible_inputs:
        output = mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")

def simulate_or():
    print("\nOR Logic Gate Simulation using MCP:")
    weights = [1, 1]
    threshold = 1
    possible_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for inputs in possible_inputs:
        output = mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")

def simulate_not():
    """
        NOT is a unary operator
    """
    print("\nNOT Logic Gate Simulation using MCP:")
    weights = [-1]
    threshold = 0
    possible_inputs = [(0, ), (1, )]

    for inputs in possible_inputs:
        output = mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs} => Output: {output}")

if __name__ == "__main__":
    simulate_and()
    simulate_or()
    simulate_not()
