print("Running")

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
# import flwr as fl
# from typing import Dict, List, Tuple
# from flwr.common import Metrics
# from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

import sys
sys.path.insert(1,'/home-mscluster/jstott/Research_Code/unet_model_serialized')
from model import unet
from my_dataLoader import load_datasets

print("IMPORTS WORKED")


from hydra.core.config_store import ConfigStore
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


from argparser import args
import argparse
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
'''
#
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,trainloader,valloader)->None:
        super().__init__()
        print("FlowerClient init_______________")
        self.trainloader = trainloader
        self.valloader = valloader
        x_train, y_train = trainloader
        imgs_shape = x_train.shape[1:]
        msks_shape = y_train.shape[1:]
        #Instatiate the model that will be trained
        self.model = get_model(imgs_shape,msks_shape)
    
    def get_parameters(self, config):
        print("get_parameters_______________") 
        param = self.model.get_weights()
        print("Param type: ",type(param))
        return param

    def fit(self, parameters, config):
        print("fit()_______________")
        print(parameters.shape)
        print(type(parameters))
        #parameters is a list of numpy arrays representing the weights of the global model
        # Copy parameters sent by the server into client's local model
        self.model.set_weights(parameters) #9:40 in video
        epochs = config['local_epochs']
        #do local training
        model_filename, model_callbacks = self.model.get_callbacks() 
        train_dataset = self.trainloader.get_tf_dataset() # Wrapped dataset for serialization
        self.model.fit(train_dataset, epochs=epochs, validation_data=self.valloader,  verbose=2, callbacks=model_callbacks)
        print("from client fit: len(self.trainloader)=",len(self.trainloader))
        return self.model.get_weights(), len(self.trainloader), {} # for sending anything (like run time or metrics) to server

    def evaluate(self, parameters, config):
        print("evaluate_______________")
        # get global model to be evaluated on client's validation data
        print(parameters.shape)
        print(type(parameters))
        self.model.set_weights(parameters)
        model_filename, model_callbacks = self.model.get_callbacks() 
        'check model.py line 76 ,81. Here we might need to add loss to the metrics so that it gets returned here'
        val_dataset = self.valloader.get_tf_dataset()
        loss, dice_coef, soft_dice_coef = self.model.evaluate(model_filename, val_dataset)

        return float(loss), len(self.valloader), {'dice_coef':dice_coef, 'soft_dice_coef':soft_dice_coef}


def get_client_fn(trainloaders, valloaders):
    print("get client fn _______________")
    #to simulate clients
    # Return a function that can be used by the VirtualClientEngine.

    # to spawn a FlowerClient with client id `cid`.
    # 
    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        print(type(trainloader=trainloaders[int(cid)]))
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            )

    return client_fn

def get_model(imgs_shape,msks_shape):
   u_net = unet()
   return u_net.create_model(imgs_shape, msks_shape, final=False) # TODO refactor FlowerClient to get trainloaders, and then get msks_shape from there

def get_evaluate_fn(testset):
    print("get_evaluated_fn_______________")

    """Return an evaluation function for server-side (i.e. centralised) evaluation."""


    # The `evaluate` function will be called after every round by the strategy
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar],):
        model = get_model()  # Construct the model
        print("About to set weights_________________")
        model.set_weights(parameters)  # Update model with the latest parameters
        print("Finished setting weights_______________")
        loss, dice_coef, soft_dice_coef = model.evaluate(testset, verbose=2)
        return loss, {"dice_coef": dice_coef,"soft_dice_coef": soft_dice_coef}
    print("about to return get_evaluated_fn")
    return evaluate


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    print("weighted average_______________")
    """Aggregation function for (federated) evaluation metrics.

    It will aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply each metric of every client by the number of examples used
    dice_coef = [num_examples * m["dice_coef"] for num_examples, m in metrics]
    soft_dice_coef = [num_examples * m["soft_dice_coef"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metrics (weighted average)
    return {
        "dice_coef": sum(dice_coef) / sum(examples),
        "soft_dice_coef": sum(soft_dice_coef) / sum(examples)
    }
'''
def main(cfg: DictConfig) -> None:
    
    print("Hello?")
    # enable_tf_gpu_growth()
    # Parse input arguments
    args = parser.parse_args()
    
    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    
    # 1. Load Data
    trainloaders, valloaders, testloader = load_datasets(cfg.num_clients, cfg.batch_size)
    print("Data Loaded")
    client1 = trainloaders[0]

    u_net = unet()
    model= u_net.create_model(client1.get_input_shape(),client1.get_output_shape() , final=False) 
    # TODO refactor FlowerClient to get trainloaders, and then get msks_shape from there
    # [x] get weights into params var
    param = model.get_weights()
    # [x] set param
    model.set_weights(param)
    # [x] fit model
    model_filename, model_callbacks = u_net.get_callbacks() 
    train_dataset = trainloaders[0]#.get_tf_dataset() # Wrapped dataset for serialization
    print("Fitting")
    model.fit(train_dataset, epochs=1, validation_data=valloaders[0],  verbose=2, callbacks=model_callbacks)
    
    
    # print("weights type")
    # print(type(model.get_weights()))
    # print(model.get_weights())
    return 0

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  # Sample 10% of available clients for training
        fraction_evaluate=1,  # Sample 5% of available clients for evaluation
        min_fit_clients=cfg.num_clients_per_round_fit ,  # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.num_clients_per_round_eval ,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            cfg.num_clients * 1
        ),  # Wait until at least n clients are available
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
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(trainloaders,valloaders),
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )
    print("Got to end of uncommented code")
    
    '''
    import matplotlib.pyplot as plt

    print(f"{history.metrics_centralized = }")

    global_accuracy_centralised = history.metrics_centralized["dice_coef"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("BRATS - IID - 2 clients with 10 clients per round")
    '''
if __name__ == "__main__":
    config_path = os.path.abspath("conf")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name="base")
        main(cfg)
