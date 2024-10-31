import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import time
 
Re = 1 #100
seed = 42
alpha = 100
beta = 100
a = 1
d = 1
 

def set_seed(seed):
    np.random.seed(seed)  # set seed for np random number generator
    torch.manual_seed(seed)  # set seed for PyTorch's random number generator on the CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # sets the seed for random number generation on a single GPU
        torch.cuda.manual_seed_all(seed)  # sets the seed for random number generation across all GPUs, ensuring reproducibility in multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VPNSFnets(nn.Module):
    def __init__(self, layer: int = 7, neurons: int = 50, act: str = 'tanh', seed: int = 42):
        # Set seed for reproducibility
        set_seed(seed)
 
        # Input layer
        super(VPNSFnets, self).__init__()
        self.linear_in = nn.Linear(4, neurons)  # (t, x, y, z)
        # Output layer
        self.linear_out = nn.Linear(neurons, 4)  # (u, v, w, p)
        # Hidden Layers
        self.layers = nn.ModuleList([nn.Linear(neurons, neurons) for i in range(layer)])
        # Activation function
        if act == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Tanh()
 
        # Apply Xavier initialization to all layers
        self._initialize_weights()
 

    def _initialize_weights(self):
        # Xavier initialization for input layer, hidden layers abd output layer
        nn.init.xavier_uniform_(self.linear_in.weight)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
 

    def forward(self, x):
        x = self.linear_in(x)
        x = self.act(x)
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        x = self.linear_out(x)
        return x
 
 

# generate training points
def generate_collocation_points(num_points, device):  # Ne
    x = (torch.rand(num_points, 1, requires_grad=True, device=device) * 2) - 1
    y = (torch.rand(num_points, 1, requires_grad=True, device=device) * 2) - 1
    z = (torch.rand(num_points, 1, requires_grad=True, device=device) * 2) - 1
    t = torch.rand(num_points, 1, requires_grad=True, device=device)
 
    return x, y, z, t
 

 
# define the exact solutions
def exact_solution(x, y, z, t, a, d):
    exp = torch.exp
    sin = torch.sin
    cos = torch.cos
 
    u = -a * (exp(a * x) * sin(a * y + d * z) + exp(a * z) * cos(a * x + d * y)) * exp(-d * d * t)
    v = -a * (exp(a * y) * sin(a * z + d * x) + exp(a * x) * cos(a * y + d * z)) * exp(-d * d * t)
    w = -a * (exp(a * z) * sin(a * x + d * y) + exp(a * y) * cos(a * z + d * x)) * exp(-d * d * t)
    p = -0.5 * a * a * (exp(2 * a * x) + exp(2 * a * y) + exp(2 * a * z) +
                        2 * sin(a * x + d * y) * cos(a * z + d * x) * exp(a * (y + z)) +
                        2 * sin(a * y + d * z) * cos(a * x + d * y) * exp(a * (z + x)) +
                        2 * sin(a * z + d * x) * cos(a * y + d * z) * exp(a * (x + y))) * exp(-2 * d * d * t)
    return u, v, w, p
 

 
def check_abnormal_values(*tensors, large_threshold=1e8, small_threshold=1e-8):
    for i, tensor in enumerate(tensors):
        tensor_np = tensor.cpu().detach().numpy()
 
        # Check for NaN values
        num_nan = np.isnan(tensor_np).sum()
        if num_nan > 0:
            print(f"Tensor {i} has {num_nan} NaN values.")
 
        # Check for infinity values
        num_inf = np.isinf(tensor_np).sum()
        if num_inf > 0:
            print(f"Tensor {i} has {num_inf} infinity values.")
 
        # Check for extremely large values
        num_large_values = (tensor_np > large_threshold).sum()
        if num_large_values > 0:
            print(f"Tensor {i} has {num_large_values} values larger than {large_threshold}.")
 
        # Check for extremely small values (excluding exact zero)
        num_small_values = ((np.abs(tensor_np) < small_threshold) & (tensor_np != 0)).sum()
        if num_small_values > 0:
            print(f"Tensor {i} has {num_small_values} values smaller than {small_threshold} but not zero.")
 


def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the models at x_f
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True)[0]
    return dy
 


# define the loss function(s)
# L = Le + alpha * Lb + beta * Li
def compute_loss_errors(x, y, z, t, x_i, y_i, z_i, t_i, x_b, y_b, z_b, t_b, t_fix, x_fix, y_fix, z_fix, model, alpha, beta, device, x_test, y_test, z_test, t_test):

    u_0, v_0, w_0, p_0 = exact_solution(x, y, z, t, a=1, d=1)
    u_pred, v_pred, w_pred, p_pred = model(torch.cat([t, x, y, z], dim=1)).split(1, dim=1)

    _, _, _, p_fix_pred = model(torch.cat([t_fix, x_fix, y_fix, z_fix], dim=1)).split(1, dim=1)
    _, _, _, p_fix = exact_solution(x_fix, y_fix, z_fix, t_fix, a, d)
    #p_fix = p_fix.view(1, 1)

 
    # test data
    u_test, v_test, w_test, p_test = exact_solution(x_test, y_test, z_test, t_test, a=1, d=1)
    u_test_pred, v_test_pred, w_test_pred, p_test_pred = model(torch.cat([t_test, x_test, y_test, z_test], dim=1)).split(1, dim=1)
    
    ones = torch.ones_like(x_b, device=device)
    neg_ones = - torch.ones_like(x_b, device=device)
 
    u_i_1, v_i_1, w_i_1, p_i_1 = exact_solution(x_i, y_i, ones, t_i, a, d)
    u_i_2, v_i_2, w_i_2, p_i_2 = exact_solution(x_i, y_i, neg_ones, t_i, a, d)
    u_i_3, v_i_3, w_i_3, p_i_3 = exact_solution(x_i, ones, z_i, t_i, a, d)
    u_i_4, v_i_4, w_i_4, p_i_4 = exact_solution(x_i, neg_ones, z_i, t_i, a, d)
    u_i_5, v_i_5, w_i_5, p_i_5 = exact_solution(ones, y_i, z_i, t_i, a, d)
    u_i_6, v_i_6, w_i_6, p_i_6 = exact_solution(neg_ones, y_i, z_i, t_i, a, d)
 
    u_b_1, v_b_1, w_b_1, p_b_1 = exact_solution(x_b, y_b, ones, t_b, a, d)
    u_b_2, v_b_2, w_b_2, p_b_2 = exact_solution(x_b, y_b, neg_ones, t_b, a, d)
    u_b_3, v_b_3, w_b_3, p_b_3 = exact_solution(x_b, ones, z_b, t_b, a, d)
    u_b_4, v_b_4, w_b_4, p_b_4 = exact_solution(x_b, neg_ones, z_b, t_b, a, d)
    u_b_5, v_b_5, w_b_5, p_b_5 = exact_solution(ones, y_b, z_b, t_b, a, d)
    u_b_6, v_b_6, w_b_6, p_b_6 = exact_solution(neg_ones, y_b, z_b, t_b, a, d)
 
    u_i_1_pred, v_i_1_pred, w_i_1_pred, p_i_1_pred = model(torch.cat([t_i, x_i, y_i, ones], dim=1)).split(1, dim=1)
    u_i_2_pred, v_i_2_pred, w_i_2_pred, p_i_2_pred = model(torch.cat([t_i, x_i, y_i, neg_ones], dim=1)).split(1, dim=1)
    u_i_3_pred, v_i_3_pred, w_i_3_pred, p_i_3_pred = model(torch.cat([t_i, x_i, ones, z_i], dim=1)).split(1, dim=1)
    u_i_4_pred, v_i_4_pred, w_i_4_pred, p_i_4_pred = model(torch.cat([t_i, x_i, neg_ones, z_i], dim=1)).split(1, dim=1)
    u_i_5_pred, v_i_5_pred, w_i_5_pred, p_i_5_pred = model(torch.cat([t_i, ones, y_i, z_i], dim=1)).split(1, dim=1)
    u_i_6_pred, v_i_6_pred, w_i_6_pred, p_i_6_pred = model(torch.cat([t_i, neg_ones, y_i, z_i], dim=1)).split(1, dim=1)
 
    u_b_1_pred, v_b_1_pred, w_b_1_pred, p_b_1_pred = model(torch.cat([t_b, x_b, y_b, ones], dim=1)).split(1, dim=1)
    u_b_2_pred, v_b_2_pred, w_b_2_pred, p_b_2_pred = model(torch.cat([t_b, x_b, y_b, neg_ones], dim=1)).split(1, dim=1)
    u_b_3_pred, v_b_3_pred, w_b_3_pred, p_b_3_pred = model(torch.cat([t_b, x_b, ones, z_b], dim=1)).split(1, dim=1)
    u_b_4_pred, v_b_4_pred, w_b_4_pred, p_b_4_pred = model(torch.cat([t_b, x_b, neg_ones, z_b], dim=1)).split(1, dim=1)
    u_b_5_pred, v_b_5_pred, w_b_5_pred, p_b_5_pred = model(torch.cat([t_b, ones, y_b, z_b], dim=1)).split(1, dim=1)
    u_b_6_pred, v_b_6_pred, w_b_6_pred, p_b_6_pred = model(torch.cat([t_b, neg_ones, y_b, z_b], dim=1)).split(1, dim=1)
 
    u_i_pred = torch.cat((u_i_1_pred, u_i_2_pred, u_i_3_pred, u_i_4_pred, u_i_5_pred, u_i_6_pred), dim=0)
    v_i_pred = torch.cat((v_i_1_pred, v_i_2_pred, v_i_3_pred, v_i_4_pred, v_i_5_pred, v_i_6_pred), dim=0)
    w_i_pred = torch.cat((w_i_1_pred, w_i_2_pred, w_i_3_pred, w_i_4_pred, w_i_5_pred, w_i_6_pred), dim=0)
    p_i_pred = torch.cat((p_i_1_pred, p_i_2_pred, p_i_3_pred, p_i_4_pred, p_i_5_pred, p_i_6_pred), dim=0)
 
    u_b_pred = torch.cat((u_b_1_pred, u_b_2_pred, u_b_3_pred, u_b_4_pred, u_b_5_pred, u_b_6_pred), dim=0)
    v_b_pred = torch.cat((v_b_1_pred, v_b_2_pred, v_b_3_pred, v_b_4_pred, v_b_5_pred, v_b_6_pred), dim=0)
    w_b_pred = torch.cat((w_b_1_pred, w_b_2_pred, w_b_3_pred, w_b_4_pred, w_b_5_pred, w_b_6_pred), dim=0)
    p_b_pred = torch.cat((p_b_1_pred, p_b_2_pred, p_b_3_pred, p_b_4_pred, p_b_5_pred, p_b_6_pred), dim=0)
 
    u_i = torch.cat((u_i_1, u_i_2, u_i_3, u_i_4, u_i_5, u_i_6), dim=0)
    v_i = torch.cat((v_i_1, v_i_2, v_i_3, v_i_4, v_i_5, v_i_6), dim=0)
    w_i = torch.cat((w_i_1, w_i_2, w_i_3, w_i_4, w_i_5, w_i_6), dim=0)
    p_i = torch.cat((p_i_1, p_i_2, p_i_3, p_i_4, p_i_5, p_i_6), dim=0)
 
    u_b = torch.cat((u_b_1, u_b_2, u_b_3, u_b_4, u_b_5, u_b_6), dim=0)
    v_b = torch.cat((v_b_1, v_b_2, v_b_3, v_b_4, v_b_5, v_b_6), dim=0)
    w_b = torch.cat((w_b_1, w_b_2, w_b_3, w_b_4, w_b_5, w_b_6), dim=0)
    p_b = torch.cat((p_b_1, p_b_2, p_b_3, p_b_4, p_b_5, p_b_6), dim=0)
 

    u_t = derivative(u_pred, t)
    u_x = derivative(u_pred, x)
    u_y = derivative(u_pred, y)
    u_z = derivative(u_pred, z)
    u_xx = derivative(u_pred, x, 2)
    u_yy = derivative(u_pred, y, 2)
    u_zz = derivative(u_pred, z, 2)
 
    v_t = derivative(v_pred, t)
    v_x = derivative(v_pred, x)
    v_y = derivative(v_pred, y)
    v_z = derivative(v_pred, z)
    v_xx = derivative(v_pred, x, 2)
    v_yy = derivative(v_pred, y, 2)
    v_zz = derivative(v_pred, z, 2)
 
    w_t = derivative(w_pred, t)
    w_x = derivative(w_pred, x)
    w_y = derivative(w_pred, y)
    w_z = derivative(w_pred, z)
    w_xx = derivative(w_pred, x, 2)
    w_yy = derivative(w_pred, y, 2)
    w_zz = derivative(w_pred, z, 2)
 
    p_x = derivative(p_pred, x)
    p_y = derivative(p_pred, y)
    p_z = derivative(p_pred, z)
 
    e_VP_1 = u_t + (u_pred * u_x + v_pred * u_y + w_pred * u_z) + p_x - 1 / Re * (u_xx + u_yy + u_zz)
    e_VP_2 = v_t + (u_pred * v_x + v_pred * v_y + w_pred * v_z) + p_y - 1 / Re * (v_xx + v_yy + v_zz)
    e_VP_3 = w_t + (u_pred * w_x + v_pred * w_y + w_pred * w_z) + p_z - 1 / Re * (w_xx + w_yy + w_zz)
    e_VP_4 = u_x + v_y + w_z
    # e_VP = e_VP_1 + e_VP_2 + e_VP_3 + e_VP_4
    e_VP = torch.cat((e_VP_1, e_VP_2, e_VP_3, e_VP_4), dim=1)
 
    # Supervised part: initial and boundary conditions
    # Unsupervised part: NS residuals
 

    # Loss
    L_e = nn.MSELoss()(e_VP, torch.zeros_like(e_VP, device=device))
 
    # uvwp_i = torch.cat((u_i, v_i, w_i, p_fix), dim=0)
    # uvwp_i_pred = torch.cat((u_i_pred, v_i_pred, w_i_pred, p_fix_pred), dim=0)
    uvw_i = torch.cat((u_i, v_i, w_i), dim=1)
    uvw_i_pred = torch.cat((u_i_pred, v_i_pred, w_i_pred), dim=1)
    uvw_b = torch.cat((u_b, v_b, w_b), dim=1)
    uvw_b_pred = torch.cat((u_b_pred, v_b_pred, w_b_pred), dim=1)
 
    L_i = nn.MSELoss()(uvw_i_pred, uvw_i)
    L_b = nn.MSELoss()(uvw_b_pred, uvw_b)

    # L_p_fix2 = torch.norm(p_fix - p_fix_pred, 2)
    L_p_fix = nn.MSELoss()(p_fix, p_fix_pred)
    # L_p_fix1 = abs(p_fix - p_fix_pred)

    loss = L_e + alpha * L_b + beta * L_i +  L_p_fix
 

    # Error
    u = torch.cat((u_0, u_i, u_b), dim=0)
    v = torch.cat((v_0, v_i, v_b), dim=0)
    w = torch.cat((w_0, w_i, w_b), dim=0)
    p = torch.cat((p_0, p_i, p_b), dim=0)
    u_pred_all = torch.cat((u_pred, u_i_pred, u_b_pred), dim=0)
    v_pred_all = torch.cat((v_pred, v_i_pred, v_b_pred), dim=0)
    w_pred_all = torch.cat((w_pred, w_i_pred, w_b_pred), dim=0)
    p_pred_all = torch.cat((p_pred, p_i_pred, p_b_pred), dim=0)
    
    # Test date error
    error_u_test = torch.norm(u_test - u_test_pred, 2) / torch.norm(u_test, 2)
    error_v_test = torch.norm(v_test - v_test_pred, 2) / torch.norm(v_test, 2)
    error_w_test = torch.norm(w_test - w_test_pred, 2) / torch.norm(w_test, 2)
    error_p_test = torch.norm(p_test - p_test_pred, 2) / torch.norm(p_test, 2)
 
    # check abnormal value
    # check_abnormal_values(u, v, w, p, u_pred_all, v_pred_all, w_pred_all, p_pred_all)
    # check_abnormal_values(u25, v25, w25, p25, u25_pred, v25_pred, w25_pred, p25_pred)
 
    #error_u = torch.norm(u - u_pred_all, 2) / torch.norm(u, 2)
    error_u = torch.norm(u - u_pred_all, 2) / torch.norm(u, 2)
    error_v = torch.norm(v - v_pred_all, 2) / torch.norm(v, 2)
    error_w = torch.norm(w - w_pred_all, 2) / torch.norm(w, 2)
    error_p = torch.norm(p - p_pred_all, 2) / torch.norm(p, 2)

 
    # return L_p_fix, L_e, L_i, L_b, loss, error_u, error_v, error_w, error_p
    return L_p_fix, L_e, L_i, L_b, loss, error_u, error_v, error_w, error_p, error_u_test, error_v_test, error_w_test, error_p_test
 
 
def train_VPNSFnets(model, epochs_adam, epochs_lbfgs, num_points, num_points_b, lbfgs_lr, layers, neurons, device, seed):
    model.to(device)
    loss_history = []
    error_u_history = []
    error_v_history = []
    error_w_history = []
    error_p_history = []
    error_u_test_history = []
    error_v_test_history = []
    error_w_test_history = []
    error_p_test_history = []
    L_e_history = []
    L_i_history = []
    L_b_history = []
    L_p_fix_history = []

    set_seed(seed)
 
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001) #0.001
    optimizer_lbfgs = optim.LBFGS(model.parameters(),
                                  lr=lbfgs_lr, 
                                  max_iter=epochs_lbfgs,
                                  history_size=100,  
                                  tolerance_grad=0,
                                  tolerance_change=0,
                                  line_search_fn="strong_wolfe")  # default: None

    optimizer = None

    x, y, z, t = generate_collocation_points(num_points, device)
    x_test, y_test, z_test, t_test = generate_collocation_points(100000, device)

    # intial conditions (t = 0)
    x_i = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    y_i = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    z_i = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    t_i = torch.zeros_like(x_i, device=device)

    # boundary conditions (t : [0---1])
    x_b = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    y_b = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    z_b = (torch.rand(num_points_b, 1, requires_grad=True, device=device) * 2) - 1
    t_b = torch.rand(num_points_b, 1, requires_grad=False, device=device)

    # p_fix
    x_fix = torch.tensor([[-1.0]], requires_grad=True, device=device)
    y_fix = torch.tensor([[-1.0]], requires_grad=True, device=device)
    z_fix = torch.tensor([[-1.0]], requires_grad=True, device=device)
    t_fix = torch.tensor([[0.0]], requires_grad=False, device=device)


    iter_temp = 0
 

    def closure():
        nonlocal iter_temp
        iter_temp += 1
        # Reset the gradients to zero before calculating gradients
        optimizer.zero_grad() 
        L_p_fix, L_e, L_i, L_b, loss, error_u, error_v, error_w, error_p, error_u_test, error_v_test, error_w_test, error_p_test \
            = compute_loss_errors(x, y, z, t, x_i, y_i, z_i, t_i, x_b, y_b, z_b, t_b, t_fix, x_fix, y_fix, z_fix, model,
                                alpha, beta, device,\
                                x_test, y_test, z_test, t_test)
        
        loss.backward(retain_graph=True)
        loss_history.append(loss.item())
        L_p_fix_history.append(L_p_fix.item())
        L_e_history.append(L_e.item())
        L_i_history.append(L_i.item())
        L_b_history.append(L_b.item())
        error_u_history.append(error_u.item())
        error_v_history.append(error_v.item())
        error_w_history.append(error_w.item())
        error_p_history.append(error_p.item())
        error_u_test_history.append(error_u_test.item())
        error_v_test_history.append(error_v_test.item())
        error_w_test_history.append(error_w_test.item())
        error_p_test_history.append(error_p_test.item())
        if iter_temp % 500 == 0:
            print(f"Epoch {iter_temp}, Loss: {loss.item()}")
            print(f"Epoch {iter_temp}, Error_p: {error_p.item()}")
            print(f"Epoch {iter_temp}, Error_u: {error_u.item()}")
            print(f"Epoch {iter_temp}, Error_p_test: {error_p_test.item()}")
            print(f"Epoch {iter_temp}, Error_u_test: {error_u_test.item()}")
        return loss
 
 
    optimizer = optimizer_adam
    for epoch in range(epochs_adam):
        optimizer_adam.step(closure)
 
    optimizer = optimizer_lbfgs
    optimizer.step(closure)
 
 
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
    plt.plot(range(len(L_p_fix_history)), L_p_fix_history, label='L_p_fix')
    plt.plot(range(len(L_e_history)), L_e_history, label='L_e')
    plt.plot(range(len(L_i_history)), L_i_history, label='L_i')
    plt.plot(range(len(L_b_history)), L_b_history, label='L_b')
    plt.xlabel('Epoch')
    plt.ylabel('Losses(Log Scale)')
    plt.yscale('log')
    plt.ylim(1e-8, 10)
    plt.title('Losses vs Epoch')
    plt.legend()
    plt.grid(True)  # Add grid to the plot
    plt.savefig(
        f'losses_epoch_log_{layers}_{neurons}_{epochs_adam}_{epochs_lbfgs}_{lbfgs_lr}_{num_points}_{num_points_b}.png')
    plt.show()
    plt.clf()  # Clear the figure after saving
 
    plt.plot(range(len(error_u_test_history)), error_u_test_history, label='Error u_test')
    plt.plot(range(len(error_u_test_history)), error_v_test_history, label='Error v_test')
    plt.plot(range(len(error_w_test_history)), error_w_test_history, label='Error w_test')
    plt.plot(range(len(error_p_test_history)), error_p_test_history, label='Error p_test')
    plt.xlabel('Epoch')
    plt.ylabel('Errors(Log Scale) at t = 0.25')
    plt.yscale('log')
    plt.ylim(1e-5, 10)
    plt.title('Test Set Errors vs Epoch')
    plt.legend()
    plt.grid(True)  # Add grid to the plot
    plt.savefig(
        f'errors_test_epoch_log_{layers}_{neurons}_{epochs_adam}_{epochs_lbfgs}_{lbfgs_lr}_{num_points}_{num_points_b}.png')
    plt.show()
    plt.clf()  # Clear the figure after saving

    plt.plot(range(len(error_u_history)), error_u_history, label='Error u')
    plt.plot(range(len(error_v_history)), error_v_history, label='Error v')
    plt.plot(range(len(error_w_history)), error_w_history, label='Error w')
    plt.plot(range(len(error_p_history)), error_p_history, label='Error p')
    plt.xlabel('Epoch')
    plt.ylabel('Errors(Log Scale)')
    plt.yscale('log')
    plt.ylim(1e-5, 10)
    plt.title('Errors vs Epoch')
    plt.legend()
    plt.grid(True)  # Add grid to the plot
    plt.savefig(
        f'errors_epoch_log_{layers}_{neurons}_{epochs_adam}_{epochs_lbfgs}_{lbfgs_lr}_{num_points}_{num_points_b}.png')
    plt.show()
    plt.clf()  # Clear the figure after saving
 
    # Printing Final Errors (Smallest in Last 100 Epochs)
    if len(error_u_history) > 100:
       min_loss = min(loss_history[-100:])
       min_error_u = min(error_u_history[-100:])
       min_error_v = min(error_v_history[-100:])
       min_error_w = min(error_w_history[-100:])
       min_error_p = min(error_p_history[-100:])
       min_error_u_test = min(error_u_test_history[-100:])
       min_error_v_test = min(error_v_test_history[-100:])
       min_error_w_test = min(error_w_test_history[-100:])
       min_error_p_test = min(error_p_test_history[-100:])
    # else:
    #    min_loss = min(loss_history)
    #    min_error_u = min(error_u_history)
    #    min_error_v = min(error_v_history)
    #    min_error_w = min(error_w_history)
    #    min_error_p = min(error_p_history)


    print(f"# of Eopchs: {len(error_p_history)}")
    print(f"Final Loss: {loss_history[-1]}")
    print(f"Final Error_u: {error_u_history[-1]}")
    print(f"Final Error_v: {error_v_history[-1]}")
    print(f"Final Error_w: {error_w_history[-1]}")
    print(f"Final Error_p: {error_p_history[-1]}")

    print(f"Final Error_u_test: {error_u_test_history[-1]}")
    print(f"Final Error_v_test: {error_v_test_history[-1]}")
    print(f"Final Error_w_test: {error_w_test_history[-1]}")
    print(f"Final Error_p_test: {error_p_test_history[-1]}") 
    # print(f"Final Error_u at t = 0.25: {error_u_t25_history[-1]}")
    # print(f"Final Error_v at t = 0.25: {error_v_t25_history[-1]}")
    # print(f"Final Error_w at t = 0.25: {error_w_t25_history[-1]}")
    # print(f"Final Error_p at t = 0.25: {error_p_t25_history[-1]}")
    
    print(f"Smallest Loss in Last 100 Epochs: {min_loss}")
    print(f"Smallest Error_u in Last 100 Epochs: {min_error_u}")
    print(f"Smallest Error_v in Last 100 Epochs: {min_error_v}")
    print(f"Smallest Error_w in Last 100 Epochs: {min_error_w}")
    print(f"Smallest Error_p in Last 100 Epochs: {min_error_p}")

    print(f"Smallest Error_u_test in Last 100 Epochs: {min_error_u_test}")
    print(f"Smallest Error_v_test in Last 100 Epochs: {min_error_v_test}")
    print(f"Smallest Error_w_test in Last 100 Epochs: {min_error_w_test}")
    print(f"Smallest Error_p_test in Last 100 Epochs: {min_error_p_test}") 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VPNSFnets()
seed = 42
epochs_adam = 30000
epochs_lbfgs = 10000
num_points = 15000
num_points_b = 31*31
lbfgs_lr = 1
neurons = 50 
layers = 7
 
time_start = time.time()
 
train_VPNSFnets(model, epochs_adam, epochs_lbfgs, num_points, num_points_b, lbfgs_lr, layers, neurons, device, seed=42)
 
time_end = time.time()

print('time cost', time_end - time_start, 's')
 
 
def plot_contour(model, num_points, num_points_b, epochs_adam, epochs_lbfgs, lbfgs_lr, layers, neurons, t, z,
                 resolution=100, device='cpu'):
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.full_like(X, t)
    Z = torch.full_like(X, z)
 
    points = torch.stack([T.flatten(), X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
 
    U, V, W, P = exact_solution(X, Y, Z, T, a, d)
 
    with torch.no_grad():
        predictions = model(points).cpu().numpy()
 
    U_pred = predictions[:, 0].reshape(resolution, resolution)
    V_pred = predictions[:, 1].reshape(resolution, resolution)
    W_pred = predictions[:, 2].reshape(resolution, resolution)
    P_pred = predictions[:, 3].reshape(resolution, resolution)
 
    # use element-wise error instead of norm
    # U_error = U_pred - U.cpu().numpy()
    # V_error = V_pred - V.cpu().numpy()
    # W_error = W_pred - W.cpu().numpy()
    # P_error = P_pred - P.cpu().numpy()
 
    # Calculate absolute errors
    U_error_abs = np.abs(U_pred - U.cpu().numpy())
    V_error_abs = np.abs(V_pred - V.cpu().numpy())
    W_error_abs = np.abs(W_pred - W.cpu().numpy())
    P_error_abs = np.abs(P_pred - P.cpu().numpy())
 
    # Calculate min and max for each variable (predicted and exact combined)
    u_min = min(U_pred.min(), U.min())
    u_max = max(U_pred.max(), U.max())
    v_min = min(V_pred.min(), V.min())
    v_max = max(V_pred.max(), V.max())
    w_min = min(W_pred.min(), W.min())
    w_max = max(W_pred.max(), W.max())
    p_min = min(P_pred.min(), P.min())
    p_max = max(P_pred.max(), P.max())
 
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
 
    ax = axes[0, 0]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), U_pred, cmap='jet', vmin=u_min, vmax=u_max)
    plt.colorbar(contour, ax=ax, label='u_pred')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u_pred at t = {t}, z = {z}')
 
    ax = axes[1, 0]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), V_pred, cmap='jet', vmin=v_min, vmax=v_max)
    plt.colorbar(contour, ax=ax, label='v_pred')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'v_pred at t = {t}, z = {z}')
 
    ax = axes[2, 0]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), W_pred, cmap='jet', vmin=w_min, vmax=w_max)
    plt.colorbar(contour, ax=ax, label='w_pred')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'w_pred at t = {t}, z = {z}')
 
    ax = axes[3, 0]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), P_pred, cmap='jet', vmin=p_min, vmax=p_max)
    plt.colorbar(contour, ax=ax, label='p_pred')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'p_pred at t = {t}, z = {z}')
 
    ax = axes[0, 1]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), U.cpu().numpy(), cmap='jet', vmin=u_min, vmax=u_max)
    plt.colorbar(contour, ax=ax, label='u_exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u_exact at t = {t}, z = {z}')
 
    ax = axes[1, 1]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), V.cpu().numpy(), cmap='jet', vmin=v_min, vmax=v_max)
    plt.colorbar(contour, ax=ax, label='v_exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'v_exact at t = {t}, z = {z}')
 
    ax = axes[2, 1]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), W.cpu().numpy(), cmap='jet', vmin=w_min, vmax=w_max)
    plt.colorbar(contour, ax=ax, label='w_exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'w_exact at t = {t}, z = {z}')
 
    ax = axes[3, 1]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), P.cpu().numpy(), cmap='jet', vmin=p_min, vmax=p_max)
    plt.colorbar(contour, ax=ax, label='p_exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'p_exact at t = {t}, z = {z}')
 
    ax = axes[0, 2]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), U_error_abs, cmap='jet')
    plt.colorbar(contour, ax=ax, label='u_error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u_error at t = {t}, z = {z}')
 
    ax = axes[1, 2]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), V_error_abs, cmap='jet')
    plt.colorbar(contour, ax=ax, label='v_error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'v_error at t = {t}, z = {z}')
 
    ax = axes[2, 2]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), W_error_abs, cmap='jet')
    plt.colorbar(contour, ax=ax, label='w_error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'w_error at t = {t}, z = {z}')
 
    ax = axes[3, 2]
    contour = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), P_error_abs, cmap='jet')
    plt.colorbar(contour, ax=ax, label='p_error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'p_error at t = {t}, z = {z}')
 
    plt.tight_layout()
    plt.savefig(
        f'contour_t{t}_z{z}_{layers}_{neurons}_{epochs_adam}_{epochs_lbfgs}_{lbfgs_lr}_{num_points}_{num_points_b}.png')
    plt.show()
 
 
plot_contour(model, num_points, num_points_b, epochs_adam, epochs_lbfgs, lbfgs_lr, layers, neurons, t=0.25, z=0, resolution=300, device=device)
