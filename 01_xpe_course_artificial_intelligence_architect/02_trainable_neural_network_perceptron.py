import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation_function(z)

    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}")
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred

                print(
                    f"Input: {xi}, Expected: {yi}, "
                    f"Predicted: {y_pred}, Error: {error}"
                )

                # Weight and bias update rule
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

                print(
                    f"Updated weights: {self.weights}, "
                    f"Bias: {self.bias}"
                )

if __name__ == "__main__":
    # Training dataset: AND truth table
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    # Create and train the perceptron
    perceptron = Perceptron(
        n_inputs=2,
        learning_rate=0.1,
        epochs=10
    )

    perceptron.train(X, y)

    # Final test
    print("\nFinal Perceptron Test (AND):")
    for xi in X:
        result = perceptron.predict(xi)
        print(f"Input: {xi} => Predicted output: {result}")

""" 
    CODE OUTPUT:
        Epoch 1
        Input: [0 0], Expected: 0, Predicted: 1, Error: -1
        Updated weights: [0. 0.], Bias: -0.1
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0. 0.], Bias: -0.1
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0. 0.], Bias: -0.1
        Input: [1 1], Expected: 1, Predicted: 0, Error: 1
        Updated weights: [0.1 0.1], Bias: 0.0
        
        Epoch 2
        Input: [0 0], Expected: 0, Predicted: 1, Error: -1
        Updated weights: [0.1 0.1], Bias: -0.1
        Input: [0 1], Expected: 0, Predicted: 1, Error: -1
        Updated weights: [0.1 0. ], Bias: -0.2
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.1 0. ], Bias: -0.2
        Input: [1 1], Expected: 1, Predicted: 0, Error: 1
        Updated weights: [0.2 0.1], Bias: -0.1
        
        Epoch 3
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.1
        Input: [0 1], Expected: 0, Predicted: 1, Error: -1
        Updated weights: [0.2 0. ], Bias: -0.2
        Input: [1 0], Expected: 0, Predicted: 1, Error: -1
        Updated weights: [0.1 0. ], Bias: -0.30000000000000004
        Input: [1 1], Expected: 1, Predicted: 0, Error: 1
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 4
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 5
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 6
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 7
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 8
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 9
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Epoch 10
        Input: [0 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [0 1], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 0], Expected: 0, Predicted: 0, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        Input: [1 1], Expected: 1, Predicted: 1, Error: 0
        Updated weights: [0.2 0.1], Bias: -0.20000000000000004
        
        Final Perceptron Test (AND):
        Input: [0 0] => Predicted output: 0
        Input: [0 1] => Predicted output: 0
        Input: [1 0] => Predicted output: 0
        Input: [1 1] => Predicted output: 1
        
        Process finished with exit code 0
"""
