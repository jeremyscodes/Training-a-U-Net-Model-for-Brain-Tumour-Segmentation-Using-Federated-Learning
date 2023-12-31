import os
import math
import argparse
from typing import Dict, List, Tuple

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import sys
sys.path.insert(1,'/home-mscluster/jstott/Research_Code/unet_model_serialized')
from model import unet
from my_dataLoader import load_datasets

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")

NUM_CLIENTS = 100
VERBOSE = 0


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val) -> None:
        # Extract input and output shapes from training data
        imgs_shape = x_train.shape[1:]
        msks_shape = y_train.shape[1:]
        # Create model
        self.model = get_model(imgs_shape,msks_shape)
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()#.numpy()?

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=32, verbose=VERBOSE
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_val, self.y_val, batch_size=64, verbose=VERBOSE
        )
        return loss, len(self.x_val), {"accuracy": acc}

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model

#The below model works (flwr is able to request initial parameters) obs not usable on our task yet
def get_dummy_model(input_shape=(28, 28, 1), num_classes=1):
    inputs = Input(input_shape)
    
    # Contracting path (Encoder)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Middle
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    # Expanding path (Decoder)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    
    return model
def get_model(imgs_shape,msks_shape):
   u_net = unet()
   return u_net.create_model(imgs_shape, msks_shape, final=False) # TODO refactor FlowerClient to get trainloaders, and then get msks_shape from there


def get_client_fn(trainloaders, valloaders):
    """Return a function to construc a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        # Extract partition for client with id = cid
        x_train, y_train = trainloaders[int(cid)]
        x_val, y_val = valloaders[int(cid)]

        # Create and return client
        return FlowerClient(x_train, y_train, x_val, y_val)
        
    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics.

    It ill aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testloader):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    print("Test Loader")

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(testloader, verbose=VERBOSE)
        return loss, {"accuracy": accuracy} # TODO change metrics

    return evaluate


def main() -> None:
    # Parse input arguments
    args = parser.parse_args()

    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    # partitions, testset = partition_mnist()
    print("Running")
    trainloaders, valloaders, testloader = load_datasets(2, 20)
    print("Loaded BRaTS")

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            NUM_CLIENTS * 0.75
        ),  # Wait until at least 75 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(testloader),  # global evaluation function
    )

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(trainloaders,valloaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()