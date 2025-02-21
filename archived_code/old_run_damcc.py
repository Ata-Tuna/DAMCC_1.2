import pickle
import torch
from torch.utils.data import DataLoader
from damcc import Damcc
from loss_functions import (
    row_wise_permutation_invariant_loss,
    row_wise_permutation_invariant_loss_batched,
    first_row_bce_loss, sinkhorn_cosine_loss, sinkhorn_row_wise_permutation_invariant_loss
)
from utils.process_graph_to_cc import CCDataset
import torch.optim as optim
import sys
import os
import argparse
from tqdm import tqdm
from colorama import Fore, Style
from datetime import datetime
import json
import time
from utils.utils import (incidence_matrix_to_graph, 
                    generate_cc_from_transposed_incidence, 
                    divide_tensors_into_lists,
                    divide_tensors)
import shutil

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument Parsing
parser = argparse.ArgumentParser(description="Training and Testing Script")
parser.add_argument('--data_name', type=str, help='name of the data for saving')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--train_file', type=str, help='Path to the training file')
parser.add_argument('--val_file', type=str, help='Path to the validation file')
parser.add_argument('--test_file', type=str, help='Path to the testing file')
parser.add_argument('--test_graphs_file', type=str, help='Path to the testing graphs file')
parser.add_argument('--model_path', type=str, help='Path to the best_val.pth model for testing')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model and results')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# Set default save directory if not provided
if args.save_dir is None:
    args.save_dir = 'experiments/last_experiment' if args.test else '/workspace/damcc/experiments'

# Create experiments directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

# Model and Dataset information for experiment folder naming
model_name = 'damcc'
dataset_name = args.data_name
experiment_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
experiment_dir = os.path.join(args.save_dir, f"test" if args.test else f"{model_name}_{dataset_name}_{experiment_time}")
os.makedirs(experiment_dir, exist_ok=True)

# Initialize model parameters
n_of_ccs = 10000

if args.train:
    with open(args.train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.val_file, 'rb') as f:
        val_data = pickle.load(f)

    train_loader = [cc for i, cc in enumerate(train_data) if i < n_of_ccs]
    val_loader = [cc for i, cc in enumerate(val_data) if i < n_of_ccs]

if args.test:
    with open(args.test_file, 'rb') as f:
        test_data = pickle.load(f)
    test_loader = [cc for i, cc in enumerate(test_data) if i < n_of_ccs]

    with open(args.test_graphs_file, 'rb') as f:
        test_graphs_data = pickle.load(f)

# # Extract basic model dimensions from the dataset
# def safe_len(data, attr, index, default=0):
#     try:
#         attribute = getattr(data[0], attr)
#         if len(attribute) > 0 and len(attribute[0]) > index:
#             return len(attribute[0][index])
#     except (IndexError, AttributeError, TypeError):
#         return default
#     return default

num_nodes = len(train_data.x_0[0]) if args.train else len(test_data.x_0[0])
n_features = len(train_data.x_0[0][1]) if args.train else len(test_data.x_0[0][1])
feature_n_0_cells = len(train_data.x_0[0][1]) if args.train else len(test_data.x_0[0][1])
feature_n_1_cells = 1
feature_n_2_cells = 1
size_g = 256

# Initialize model and optimizer
model = Damcc(num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_function = sinkhorn_cosine_loss

# ASCII banner
ascii_banner = """
                                                                                
                          )                           (           )       (     
 (                     ( /(               )           )\   (   ( /(   (   )\ )  
 )\   `  )    (    (   )\())   (   (     (     `  )  ((_) ))\  )\()) ))\ (()/(  
((_)  /(/(    )\   )\ ((_)\    )\  )\    )\  ' /(/(   _  /((_)(_))/ /((_) ((_)) 
| __|((_)_\  ((_) ((_)| |(_)  ((_)((_) _((_)) ((_)_\ | |(_))  | |_ (_))   _| |  
| _| | '_ \)/ _ \/ _| | ' \  / _|/ _ \| '  \()| '_ \)| |/ -_) |  _|/ -_)/ _` |  
|___|| .__/ \___/\__| |_||_| \__|\___/|_|_|_| | .__/ |_|\___|  \__|\___|\__,_|  
     |_|                                      |_|                               
"""

# Early stopping and learning rate reduction
best_val_loss = float('inf')
early_stop_counter = 0
lr_reduction_counter = 0
lr_reduction_patience = 10  # Reduce LR if no improvement after 10 epochs
early_stop_patience = 20    # Stop training if no improvement after 20 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_reduction_patience, factor=0.5, verbose=True)
one_cells_weight = 0.5

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=args.max_epochs):
    global best_val_loss, early_stop_counter
    total_train_loss = []
    total_val_loss = []
    start_time = time.time()

    # Variables to track the best model
    best_epoch = -1
    best_model_state = None

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0

            for cc in tqdm(train_loader):
                print(f"{Fore.GREEN}{Style.BRIGHT}At Epoch {epoch+1}:")
                x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t = [tensor.to(device) for tensor in cc]

                # Convert adjacency matrices to Float
                a1 = a1.float()
                a2 = a2.float()
                coa2 = coa2.float()
                b1 = b1.float()
                b2 = b2.float()

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)

                print("ONE FORWARD PASS DONE")

                # Compute loss
                loss_1 = loss_function(sampled_b10, b10_t)
                loss_2 = loss_function(sampled_b20, b20_t)
                loss = (one_cells_weight*loss_1 + (1-one_cells_weight)*loss_2)

                if isinstance(loss, torch.Tensor):
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                else:
                    print("Loss is not a tensor; skipping backward pass.")
            
                print("ONE BACKWARD PASS DONE")

            avg_train_loss = epoch_train_loss / len(train_loader)
            total_train_loss.append(avg_train_loss)

            # Validation phase
            avg_val_loss = validate_model(model, val_loader)
            total_val_loss.append(avg_val_loss)

            # Check if current model is the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict()  # Save the best model state
                early_stop_counter = 0
                # Save best model state to file
                torch.save(best_model_state, os.path.join(experiment_dir, 'best_val_model.pth'))
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered!")
                    break

            # Adjust learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save losses and model state after each epoch
            epoch_data = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss
            }
            with open(os.path.join(experiment_dir, 'epoch_losses.json'), 'a') as f:
                json.dump(epoch_data, f)
                f.write('\n')

            print(ascii_banner)
            print(f"{Fore.GREEN}{Style.BRIGHT}Epoch {epoch+1}:")
            print(f"🔥 Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}{Style.RESET_ALL}")

            if early_stop_counter >= early_stop_patience:
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model and exiting...")
        save_training_state(total_train_loss, total_val_loss, start_time, experiment_dir, best_epoch, best_model_state)
        sys.exit()

    # Save training logs after normal completion
    save_training_state(total_train_loss, total_val_loss, start_time, experiment_dir, best_epoch, best_model_state)

    # Copy experiment_dir and rename it to last_experiment
    last_experiment_dir = os.path.join(args.save_dir, 'last_experiment')
    if os.path.exists(last_experiment_dir):
        shutil.rmtree(last_experiment_dir)  # Remove existing last_experiment directory if it exists
    shutil.copytree(experiment_dir, last_experiment_dir)
    print(f"Copied {experiment_dir} to {last_experiment_dir}")

def save_training_state(train_loss, val_loss, start_time, experiment_dir, best_epoch, best_model_state):
    end_time = time.time()
    training_data = {
        "total_training_time": end_time - start_time,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_validation_epoch": best_epoch
    }
    with open(os.path.join(experiment_dir, 'training_data.json'), 'w') as f:
        json.dump(training_data, f)

# Validation loop
def validate_model(model, val_loader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for cc in val_loader:
            x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t = [tensor.to(device) for tensor in cc]

            # Convert adjacency matrices to Float
            a1 = a1.float()
            a2 = a2.float()
            coa2 = coa2.float()
            b1 = b1.float()
            b2 = b2.float()

            sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)

            # Compute the loss
            loss_1 = loss_function(sampled_b10, b10_t)
            loss_2 = loss_function(sampled_b20, b20_t)
            loss = (one_cells_weight*loss_1 + (1-one_cells_weight)*loss_2)

            # Check if the loss is a tensor (non-empty) or a float (empty tensors handled by returning 0.0)
            if isinstance(loss, torch.Tensor):
                total_val_loss += loss.item()
            else:
                total_val_loss += loss  # Here loss is already a float, no need to call .item()

    return total_val_loss / len(val_loader)

# Test loop
def test_model(model, test_loader, model_path, save_path, test_graphs_data):
    model.eval()
    model.load_state_dict(torch.load(model_path))  # Load the specific model for testing
    total_test_loss = 0
    sampled_b10_list = []
    sampled_b20_list = []
    target_b10_list = []
    target_b20_list = []
    sampled_graphs = []
    sampled_ccs = []

    try:
        with torch.no_grad():
            for cc in test_loader:
                x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t  = [tensor.to(device) for tensor in cc]

                # Convert adjacency matrices to Float
                a1 = a1.float()
                a2 = a2.float()
                coa2 = coa2.float()
                b1 = b1.float()
                b2 = b2.float()

                sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)

                # Remove duplicate rows
                sampled_b20 = torch.unique(sampled_b20, dim=0)
                sampled_b10 = torch.unique(sampled_b10, dim=0)

                sampled_b10 = sampled_b10.cpu().numpy()
                sampled_b20 = sampled_b20.cpu().numpy()
                sampled_b10_list.append(sampled_b10)  # Convert tensors to numpy for saving
                sampled_b20_list.append(sampled_b20)
                target_b10_list.append(b10_t)  # Convert tensors to numpy for saving
                target_b20_list.append(b20_t)
                sampled_graphs.append(incidence_matrix_to_graph(sampled_b10))
                sampled_ccs.append(generate_cc_from_transposed_incidence(sampled_b10, sampled_b20))

        # bring to the same shape as test data. This chunk is done for evaluation 
        if len(sampled_graphs) % len(test_graphs_data) != 0:
            raise ValueError("The number of sampled graphs must be divisible by the length of the test graphs data.")
        if len(sampled_ccs) % len(test_graphs_data) != 0:
            raise ValueError("The number of sampled CCs must be divisible by the length of the test graphs data.")
        sampled_graphs = divide_tensors_into_lists(test_graphs_data, sampled_graphs)
        sampled_ccs = divide_tensors_into_lists(test_graphs_data, sampled_ccs)
        sampled_b10s = divide_tensors(test_graphs_data, sampled_b10_list, sample=True)
        sampled_b20s = divide_tensors(test_graphs_data, sampled_b20_list, sample=True)
        target_b10s = divide_tensors(test_graphs_data, target_b10_list, sample=False)
        target_b20s = divide_tensors(test_graphs_data, target_b20_list, sample=False)

        # Save generated lists
        with open(os.path.join(save_path, 'target_b10s.pkl'), 'wb') as f:
            pickle.dump(target_b10s, f)
        with open(os.path.join(save_path, 'target_b20s.pkl'), 'wb') as f:
            pickle.dump(target_b20s, f)

        with open(os.path.join(save_path, 'sampled_b10s.pkl'), 'wb') as f:
            pickle.dump(sampled_b10s, f)
        with open(os.path.join(save_path, 'sampled_b20s.pkl'), 'wb') as f:
            pickle.dump(sampled_b20s, f)

        with open(os.path.join(save_path, 'sampled_graphs.pkl'), 'wb') as f:
            pickle.dump(sampled_graphs, f)
        with open(os.path.join(save_path, 'sampled_ccs.pkl'), 'wb') as f:
            pickle.dump(sampled_ccs, f)

    except KeyboardInterrupt:
        print("Testing interrupted by user. Saving current state and exiting...")
        # Save the current state of the results
        with open(os.path.join(save_path, 'interrupted_sampled_b10_list.pkl'), 'wb') as f:
            pickle.dump(sampled_b10_list, f)
        with open(os.path.join(save_path, 'interrupted_sampled_b20_list.pkl'), 'wb') as f:
            pickle.dump(sampled_b20_list, f)
        with open(os.path.join(save_path, 'interrupted_sampled_graphs.pkl'), 'wb') as f:
            pickle.dump(sampled_graphs, f)
        with open(os.path.join(save_path, 'interrupted_sampled_ccs.pkl'), 'wb') as f:
            pickle.dump(sampled_ccs, f)
        print("Test results saved. Exiting gracefully.")
        sys.exit()

# Execute Training or Testing
if args.train:
    train_model(model, train_loader, val_loader)
elif args.test:
    if args.model_path is None:
        raise ValueError("Model path must be provided for testing.")
    test_model(model, test_loader, args.model_path, experiment_dir, test_graphs_data)
