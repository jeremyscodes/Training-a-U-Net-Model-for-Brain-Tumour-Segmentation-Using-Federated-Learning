import flwr as fl
import tensorflow as tf
import segmentation_models as sm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.io import loadmat
import numpy as np
import hydra
from typing import List,Tuple,Dict
Metrics = Dict[str,float]

'''with open("server_ip.txt", "r") as f:
    server_ip = f.read().strip()

server_address = f"{server_ip}:8080"'''


# Unet model (client)
BACKBONE = 'resnet34'

from keras.layers import Input, Conv2D
from keras.models import Model

#base_model = sm.Unet(BACKBONE, encoder_weights=None)#or pretrained on seg task (cityscape) with imagenet and without
#N=1
#inp = Input(shape=(None,None, N))
#l1 = Conv2D(3, (1,1))(inp)
#out = base_model(l1)
#model = Model(inp, out, name=base_model.name)
# GET DATASET

data = pd.read_csv('../patient_files.csv')

# Create a GroupShuffleSplit instance
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42) #note 10% testing

# Get the train and test indices
train_val_indices, test_indices = next(gss.split(data, groups=data['PatientID']))

# Split the train_val data into training and validation sets
train_indices, val_indices = next(gss.split(data.iloc[train_val_indices], groups=data.iloc[train_val_indices]['PatientID']))

# Create lists of file names for training, validation and test sets
train_files = data.iloc[train_val_indices].iloc[train_indices]['FileName'].tolist()
val_files = data.iloc[train_val_indices].iloc[val_indices]['FileName'].tolist()
test_files = data.iloc[test_indices]['FileName'].tolist()

BATCH_SIZE = 32
NUM_CLIENTS = 2

#Trying a direct load without a datagenerator (might work on cluster)
def load_data(file_names):
    # Initialize lists to store data
    images = []
    masks = []

    # Loop over all .mat files
    for fname in file_names:
        # Load the mat file
        mat_data = loadmat('../Figshare/brainTumorDataPublic_convert_All/'+ fname)
        image = mat_data['image']
        mask = mat_data['mask']
        # Append data to lists
        images.append(image)
        masks.append(mask)
    return images, masks

def load_datasets():
    # Convert test data to tf.data.Dataset
    x_test, y_test = load_data(test_files)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Group training files by PatientID and filter by train_files
    grouped_data = data[data['FileName'].isin(train_files)]
    grouped_train_files = grouped_data.groupby('PatientID').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    
    shuffled_patient_groups = grouped_train_files['PatientID'].unique()
    np.random.shuffle(shuffled_patient_groups)

    # Allocate entire patient groups to individual clients
    client_patient_groups = np.array_split(shuffled_patient_groups, NUM_CLIENTS)
    client_dataframes = [grouped_train_files[grouped_train_files['PatientID'].isin(group)] for group in client_patient_groups]

    # Create DataLoaders for each client
    trainloaders = []
    valloaders = []
    for client_df in client_dataframes:
        x_client, y_client = load_data(client_df['FileName'].tolist())
        ds = tf.data.Dataset.from_tensor_slices((x_client, y_client))
        ds_size = len(x_client)
        len_val = ds_size // 10  # 10 % validation set
        len_train = ds_size - len_val
        ds_train = ds.skip(len_val).take(len_train).batch(BATCH_SIZE).shuffle(buffer_size=1000)
        ds_val = ds.take(len_val).batch(BATCH_SIZE)
        trainloaders.append(ds_train)
        valloaders.append(ds_val)

    testloader = test_dataset.batch(BATCH_SIZE)
    return trainloaders, valloaders, testloader



# CREATE CLIENT CLASS

class UnetClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader) ->None:
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_weights(self):
        return self.model.get_weights()
    
    def fit(self, weights, config):
        self.model.set_weights(weights)
        self.model.fit(self.train_loader, epochs=1, batch_size=32, steps_per_epoch=3)  # Remove `steps_per_epoch=3` to train on the full dataset. 3 batches per epoch
        return self.model.get_weights(), len(self.train_loader),{}
    
    def evaluate(self, weights, config):
        self.model.set_weights(weights)
        loss, iou_score, f1_score = self.model.evaluate(self.val_loader)
        return loss, len(self.val_loader), iou_score, f1_score

#The data in trainloaders[0] is the training dataset for the first client.

with open("server_ip.txt", "r") as f:
    server_ip = f.read().strip()

s_address = f"{server_ip}:8080"

#fl.client.start_numpy_client(server_address = s_address, client = UnetClient(model,trainloaders,valloaders))
#from mscluster: 10.100.14.48

def client_fn(cid: str) -> fl.client.NumPyClient:
    """Create a Flower client representing a single organization."""
    preprocess_input = sm.get_preprocessing(BACKBONE)
    base_model = sm.Unet(BACKBONE, encoder_weights=None)#or pretrained on seg task (cityscape) with imagenet and without
    base_model = sm.Unet(BACKBONE, encoder_weights=None)#or pretrained on seg task (cityscape) with imagenet and without
    N=1
    inp = Input(shape=(None,None, N))
    l1 = Conv2D(3, (1,1))(inp)
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)

    model.compile(
        'Adam',
        loss=sm.losses.dice_loss, # changed from bce_jaccard_loss
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)], #f1 is Dice score
            #changed on Saturday to use thresholding function. not run yet. Only running when i have wandb set up
    )

    trainloaders, valloaders, testloader = load_datasets()
    
    # Get the data loaders specific to the client with ID = cid
    train_loader = trainloaders[int(cid)]
    val_loader = valloaders[int(cid)]

    # Create and return a UnetClient instance
    return UnetClient(model, train_loader)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    iou_scores = [num_examples * m["iou_score"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "iou_score": sum(iou_scores) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples)
    }

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 10 clients for training
    min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
    min_available_clients=2,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if tf.config.experimental.list_physical_devices('GPU'):
    client_resources = {"num_gpus": 1}


# Start simulation

def main() -> None:

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    main()




