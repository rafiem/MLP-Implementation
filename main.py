from mlp import MLP 

filename      = "iris.csv"
test_size     = 50
learning_rate = 0.1
seed          = 5
epoch         = 275
theta_input   = [[0.4, 0.6, 0.2, 0.55], [0.6, 0.3, 0.45, 0.55], [0.35, 0.75, 0.45, 0.4], [0.55, 0.8, 0.25, 0.2]]
theta_hidden  = [[0.4, 0.25, 0.7, 0.35], [0.65, 0.40, 0.3, 0.7]]
bias_input    = [0.5, 0.6, 0.4, 0.3]
bias_hidden   = [0.55, 0.45]

mlp = MLP(learning_rate, filename, test_size, seed)
mlp.set_thetas(theta_input, theta_hidden)
mlp.set_bias(bias_input, bias_hidden)

for i in range(epoch):
  mlp.training()
  mlp.testing()

mlp.plotting() 