## import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import statistics
from datetime import datetime
from sklearn.model_selection import train_test_split
import random, sys, copy, os, json

## helper functions

# given a row of sessions, take domain_ids and domain_scores, which are in string format separated by ",", and replace with a list of the values
def process_row(row):
    values_a = [int(x.strip()) for x in str(row['domain_ids']).split(',')]
    values_b = [float(x.strip()) for x in str(row['domain_scores']).split(',')]
    return values_a, values_b

# take in a dataframe of a patient's session, extract information useful for training
def create_training_data(data: pd.DataFrame):
    # Initialize variables
    session_row = [] # contents of a row (patient id, encoding, cur score, prev score, repeat)
    overall = [] # aggregate of everything (n sessions x 44)

    cur_score = np.zeros((14)) # score for each session
    prev_score = None

    seen = {} # dictionary for seen
    patient_id = data["patient_id"].iloc[0] # save patient_id

    # Sort data by session start time
    data = data.sort_values(by=["start_time_min"])

    # Process each row
    for idx, row in data.iterrows():
        domains, domain_scores = process_row(row)  # returns a list of domains : int and of domain_scores : float

        # Track repeat status and update scores
        repeat = False

        for j, domain in enumerate(domains):
            if domain not in seen:
                seen[domain] = True
            else:
                repeat = True

            cur_score[domain - 1] = domain_scores[j] # update score in the loop

        # Encode domains for this session
        domain_encoding = np.zeros(14)
        for domain in domains:
            domain_encoding[domain - 1] = 1
        
        

        # if the session does not contain the target domain or is the first (no prev score), continue in the loop without doing anything, do this before appending
        if prev_score is None:
            session_row = []
            prev_score = cur_score.copy()
            continue
        # assert np.sum(domain_encoding) != 1, "continue not working"

        # append everything in the row list
        session_row.append(patient_id)
        session_row.extend(domain_encoding.copy().tolist())
        session_row.extend(prev_score.copy().tolist())
        session_row.extend(cur_score.copy().tolist())
        session_row.append(repeat)
        assert len(session_row) == 44, "session row length weird"

        # append row to overall, reset
        overall.append(session_row)
        session_row = []
        prev_score = cur_score.copy()

    # Convert to numpy arrays
    if overall:
        overall = np.array(overall)
        assert len(overall.shape) == 2, "dimensions of overall wrong"
    else:
        # Handle case where scores is empty
        return pd.DataFrame(columns=["patient_id"] + ["domain %d encoding" % i for i in range(1, 15)] +
                                   ["domain %d score" % i for i in range(1, 15)] +
                                   ["domain %d target" % i for i in range(1, 15)] +
                                   ["repeat"])
    
        # Create column names
    column_names = (
        ["patient_id"]
        + [f"domain {i} encoding" for i in range(1, 15)]
        + [f"domain {i} score" for i in range(1, 15)]
        + [f"domain {i} target" for i in range(1, 15)]
        + ["repeat"]
    )

    # Create dataframe
    scores_df = pd.DataFrame(overall, columns=column_names)
    scores_df.reset_index(drop=True, inplace=True)
    return scores_df

# create missing indicator when given the score data
def create_missing_indicator(data):
    # Set seeds for reproducibility
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    (l, w) = data.shape
    temp = np.zeros((l, w*2))
    for i in range(l):
        for d in range(w):
            p = data[i, d]
            # update output array
            if p == 0:
                missing_ind = np.random.choice(2, 1)[0]
                temp[i, d*2] = missing_ind
                temp[i, d*2+1] = missing_ind
            else:
                temp[i, d*2] = p # score
                temp[i, d*2+1] = 1-p # 1-score
    return copy.deepcopy(temp)

# process data, takes in filename that contains the data we are using and number of samples we want to use this run, returns train and test dataframes
def data_process(filename : str, n_samples : int):
    # read in filtered sessions data
    df = pd.read_csv(filename)
    # sort dataframe by start_time_min
    df["start_time_min"] = df["start_time_min"].astype('datetime64[ns]')
    df = df.sort_values(by=["patient_id", "start_time_min"])
    # create training data using df
    data = df.groupby("patient_id")[df.columns].apply(create_training_data).reset_index(drop=True)
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    ## one sample for train, only to see if it learns that one example
    train_data = train_data[:n_samples].copy()
    test_data = test_data[:n_samples].copy()
    return train_data, test_data

# given a processed dataframe, return data and target tensors that can be put in the model
def create_model_data(data : pd.DataFrame):
    target = data[target_columns].copy().to_numpy() * data[encoding_columns].copy().to_numpy()
    data_scores = create_missing_indicator(data[score_columns].copy().to_numpy())
    final_data = np.hstack((data[encoding_columns].copy().to_numpy(), data_scores))
    return torch.from_numpy(final_data).float().to(device), torch.from_numpy(target).float().to(device)

# plot average improvement plots and store, d_type= Ground Truth or Prediction, mode=train or test, cur_score=whatever we need, data=test or train data
def plot_average_improvements(d_type, mode, cur_score, data):
    # Step 1: Compute differences
    differences = cur_score - data[score_columns].copy().to_numpy()
    # Step 2: Mask the differences using the encoding array
    masked_differences = np.where(data[encoding_columns].copy().to_numpy() == 1, differences, 0)  # Retain differences only where encoding is 1
    # Step 3: Compute the column-wise sum and count
    column_sums = np.sum(masked_differences, axis=0)  # Sum of differences for each column
    column_counts = np.sum(data[encoding_columns].copy().to_numpy(), axis=0)          # Number of 1s in each column
    # Step 4: Filter out columns with no encoding == 1
    valid_columns = column_counts > 0  # Boolean mask for valid columns
    filtered_sums = column_sums[valid_columns]
    filtered_counts = column_counts[valid_columns]
    # Step 5: Compute the column-wise averages for valid columns
    filtered_averages = filtered_sums / filtered_counts
    filtered_column_indices = np.where(valid_columns)[0]
    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))  # Create the figure and axes
    bars = ax.bar(range(len(filtered_averages)), filtered_averages, tick_label=[f"{i+1}" for i in filtered_column_indices])
    # Add values to the bars
    ax.bar_label(bars, fmt='%.4f', label_type='edge')
    # Set the y-axis range
    ax.set_ylim(-0.1, 0.1)
    # Add labels and title
    title_s = "%s %s Data Domain Improvement Averages" % (d_type, mode)
    plt.xlabel("Domains", fontsize=12)
    plt.ylabel("Average Difference", fontsize=12)
    plt.title(title_s, fontsize=16)
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_dir + title_s + ".png")

## input : 14 domain encodings + 14 domains (28 total features with missing indicator)
## output: 28 score (prediction for the scores after next domain)
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_domains = 14
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_domains * 3, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, n_domains)
        )

    def forward(self, x):
        return self.model(x)

# used for batch training
class customDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.target[index, :]

# train the model
def train_model(x_train, x_val,y_train, y_val, epochs, model, optimizer, loss_function, batch_size, history):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    losses = []
    val_losses = []
    w = 14 ## hardcoded

    data_set = customDataset(x_train, y_train)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        if epoch % 1000 == 0: print("epoch %d" % epoch)
        epoch_loss = []
        ## training
        model.train()
        for batch_x, batch_y in data_loader:
        # Output of Autoencoder
            reconstructed = model(batch_x)
            
            # Calculating the loss function
            loss = loss_function(reconstructed, batch_y.reshape(reconstructed.shape))

            optimizer.zero_grad()
            loss.backward()

            # store history of weights, bias, gradients in the dictionary history
            history["weight"].append(model.model[0].weight.clone().cpu().detach().numpy().tolist())
            history["bias"].append(model.model[0].bias.clone().cpu().detach().numpy().tolist())
            history["gradient"].append(model.model[0].weight.grad.clone().cpu().numpy().tolist())

            optimizer.step()
            
            # Storing the losses in a list for plotting
            epoch_loss.append(loss.clone().cpu().item())

        losses.append(statistics.mean(epoch_loss))

        ## validation
        model.eval()
        with torch.no_grad():
            val_rs = x_val.reshape(-1, w * 3)
            val_t = val_rs.clone().detach().type(torch.float32)
            answer = model(val_t)
            val_loss = loss_function(answer, y_val.reshape(answer.shape))
        val_losses.append(val_loss.clone().cpu())
    return losses, val_losses, history, model

# return metrics from model training
def train(lr, epochs, batch_size, train_data):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    losses_2d = []
    val_losses_2d = []
    model_history = {"weight": [], "bias": [], "gradient": []}

    # model related
    model = NN().to(device)
    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # x_train, x_val, y_train, y_val = train_test_split(train_data, target, test_size=0.50)
    data, target = create_model_data(train_data)
    x_train = data.clone().to(device)
    y_train = target.clone().to(device)

    x_val = data.clone().to(device)
    y_val = target.clone().to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(x_train)
        zero_loss = loss_function(predictions, y_train)

        predictions = model(x_val)
        zero_loss_val = loss_function(predictions, y_val.reshape(predictions.shape))
    
    losses, val_losses, model_history, model = train_model(x_train, x_val, y_train, y_val, epochs, model, optimizer, loss_function, batch_size, model_history)
    losses = [zero_loss.item()] + losses
    val_losses = [zero_loss_val.item()] + val_losses
    
    losses_2d.append(losses)
    val_losses_2d.append(val_losses)

    # plt training curve
    plot_curve(losses_2d, val_losses_2d)
    return model, model_history


def plot_mean_and_std(data, color_choice, setting=""):
    # Convert data to a NumPy array for easier manipulation
    data_array = np.array(data)
    
    # Calculate mean and standard deviation
    means = np.mean(data_array, axis=0)
    stds = np.std(data_array, axis=0)
    
    # Create the x-axis values
    x_values = np.arange(len(means))
    
    # Plotting
    plt.plot(x_values, means, label='%s Mean' % setting, color=color_choice)  # Mean line
    plt.fill_between(x_values, means - stds, means + stds, color=color_choice, alpha=0.2, label='%s Standard Deviation' % setting)
    
    plt.title('Mean and Standard Deviation Plot')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)


def plot_curve(train_loss, val_loss):
    with torch.no_grad():
        plt.figure()
        plot_mean_and_std(train_loss, "blue", "Training")
        plot_mean_and_std(val_loss, "orange", "Validation")
        plt.savefig(output_dir + "curve.png")

# return predictions, loss, and mae
def predict(model, data):
    x, y = create_model_data(data)
    with torch.no_grad():
        predictions = model(x)
        loss = loss_function(predictions, y.reshape(predictions.shape))    
        return predictions.clone().cpu().numpy(), loss.clone().cpu().item(), torch.mean(torch.abs(predictions - y.reshape(predictions.shape))).clone().cpu().item()


## main code
print("set up")
# process system arguments and set global variables
output_dir = sys.argv[1]
data_source = "data/filtered_ds.csv"
# grid search hyper parameters
n_samples = 10
learning_rate = 1e-3
n_epochs = int(1e2)
batch_sizes = int(1)
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
# column names
score_columns = ["domain %d score" %i for i in range(1, 15)]
encoding_columns = ["domain %d encoding" %i for i in range(1, 15)]
target_columns = ["domain %d target" %i for i in range(1, 15)]
repeat_columns = ["repeat"]
# model records
records = dict()

# initialize random seed and cuda device
# Set seeds for reproducibility
rand_seed = 42
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)

# Ensure deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# make sure GPU is used
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise Exception("No GPU Available")

# create output directory to store this run's results
try:
    os.mkdir(output_dir)
except Exception as e:
    print("Error occurred creating directory")

print("data processing")
train_data, test_data = data_process(data_source, n_samples)
print("training model")
model, model_history = train(learning_rate, n_epochs, batch_sizes, train_data)

print("plotting results")
prediction, loss, mae = predict(model, train_data)
plot_average_improvements("Ground Truth", "Train", train_data[target_columns].copy().to_numpy() * train_data[encoding_columns].copy().to_numpy(), train_data)
plot_average_improvements("Prediction", "Train", prediction, train_data)
records["Train"] = (loss.clone().cpu().numpy().tolist(), mae.clone().cpu().numpy().tolist())

prediction, loss, mae = predict(model, test_data)
plot_average_improvements("Ground Truth", "Test", test_data[target_columns].copy().to_numpy() * test_data[encoding_columns].copy().to_numpy(), test_data)
plot_average_improvements("Prediction", "Test", prediction, test_data)
records["Test"] = (loss.clone().cpu().numpy().tolist(), mae.clone().cpu().numpy().tolist())

# save records in a json file
with open(output_dir+"records.json", "w") as f:
    json.dump(records, f)
# save model history in a json file
with open(output_dir+"history.json", "w") as f:
    json.dump(model_history, f)

print("done")