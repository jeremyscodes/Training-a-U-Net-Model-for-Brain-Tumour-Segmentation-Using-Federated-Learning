print("Running")
from collections import defaultdict
import string
import matplotlib.pyplot as plt
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import flwr as fl
import tensorflow as tf
import psutil
from flwr.server import Server
from typing import Dict, List, Tuple
from flwr.common import Metrics, parameters_to_ndarrays
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
import numpy as np
import concurrent.futures
import timeit
from logging import DEBUG, INFO
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from typing import Dict, List, Optional, Tuple, Union

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


from flwr.server import Server as FlowerServer

import sys
from ang_loader import load_datasets #using new loader (serializable)
from copy import deepcopy


from hydra.core.config_store import ConfigStore
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


# from argparser import args
import argparse
parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

from tensorflow import keras as K

class unet(object):
    """
    2D U-Net model class
    """

    def __init__(self, channels_first=False,
                 fms=16,#Change to 32 for 80% DC
                 output_path=os.path.join("./output/"),
                 inference_filename="2d_unet_decathlon",
                 blocktime=0,
                 num_threads=min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False)),
                 learning_rate=0.0001,
                 weight_dice_loss=0.85,
                 num_inter_threads=1,
                 use_upsampling=False,
                 use_dropout=True,
                 print_model=False):
    # def __init__(self, channels_first=settings.CHANNELS_FIRST,
    #              fms=settings.FEATURE_MAPS,
    #              output_path=settings.OUT_PATH,
    #              inference_filename=settings.INFERENCE_FILENAME,
    #              blocktime=settings.BLOCKTIME,
    #              num_threads=settings.NUM_INTRA_THREADS,
    #              learning_rate=settings.LEARNING_RATE,
    #              weight_dice_loss=settings.WEIGHT_DICE_LOSS,
    #              num_inter_threads=settings.NUM_INTRA_THREADS,
    #              use_upsampling=settings.USE_UPSAMPLING,
    #              use_dropout=settings.USE_DROPOUT,
    #              print_model=settings.PRINT_MODEL):
        self.channels_first = channels_first
        if self.channels_first:
            """
            Use NCHW format for data
            """
            self.concat_axis = 1
            self.data_format = "channels_first"

        else:
            """
            Use NHWC format for data
                N: Number of images in the batch
                H: Height of the image
                W: Width of the image
                C: Number of channels
            """
            self.concat_axis = -1
            self.data_format = "channels_last"

        self.fms = fms  # 32 or 16 depending on your memory size

        self.learningrate = learning_rate
        self.weight_dice_loss = weight_dice_loss

        print("Data format = " + self.data_format)
        K.backend.set_image_data_format(self.data_format)

        self.output_path = output_path
        self.inference_filename = inference_filename

        self.metrics = [self.dice_coef, self.soft_dice_coef, self.iou_coef, self.precision, self.accuracy, self.specificity]

        self.loss = self.dice_coef_loss
        # self.loss = self.combined_dice_ce_loss

        self.optimizer = K.optimizers.Adam(learning_rate=self.learningrate)

        self.custom_objects = {
            "combined_dice_ce_loss": self.combined_dice_ce_loss,
            "dice_coef_loss": self.dice_coef_loss,
            "dice_coef": self.dice_coef,
            "soft_dice_coef": self.soft_dice_coef,
            "iou_coef": self.iou_coef,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "specificity": self.specificity}

        self.blocktime = blocktime
        self.num_threads = num_threads
        self.num_inter_threads = num_inter_threads

        self.use_upsampling = use_upsampling
        self.use_dropout = use_dropout
        self.print_model = print_model

    def dice_coef(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)
    
    def iou_coef(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Intersection over Union (IoU)
        \frac{\text{Intersection}}{\text{Union}} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
        where T is the ground truth mask and P is the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis) - intersection
        coef = (intersection + smooth) / (union + smooth)

        return tf.reduce_mean(coef)

    def soft_dice_coef(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Sorenson (Soft) Dice  - Don't round the predictions
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)
    
    def specificity(self, y_true, y_pred):
        """
        Compute specificity.
        """
        # Threshold predictions
        y_pred = K.backend.round(y_pred)
        
        # Count true negatives
        tn = K.backend.sum(K.backend.round(K.backend.clip((1-y_true) * (1-y_pred), 0, 1)))
        
        # Count false positives
        fp = K.backend.sum(K.backend.round(K.backend.clip((1-y_true) * y_pred, 0, 1)))
        
        # Compute specificity
        specificity = tn / (tn + fp + K.backend.epsilon())
        return specificity

    def precision(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Precision
        \frac{\left | T \right | \cap \left | P \right |}{\left | P \right |}
        where T is the ground truth mask and P is the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        true_positives = tf.reduce_sum(target * prediction, axis=axis)
        predicted_positives = tf.reduce_sum(prediction, axis=axis)
        numerator = true_positives + smooth
        denominator = predicted_positives + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)


    def accuracy(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Accuracy
        \frac{\left | T \right | \cap \left | P \right | + \left | T \right | \cap \left | \neg P \right |}{\text{Total Predictions}}
        where T is the ground truth mask, P is the prediction mask, and \neg P is the negation of the prediction mask
        """
        prediction = K.backend.round(prediction)  # Round to 0 or 1

        true_positives = tf.reduce_sum(target * prediction, axis=axis)
        true_negatives = tf.reduce_sum((1 - target) * (1 - prediction), axis=axis)
        total_predictions = tf.reduce_prod(tf.shape(target)[1:])
        
        numerator = true_positives + true_negatives + smooth
        denominator = tf.cast(total_predictions, tf.float32) + smooth
        coef = numerator / denominator

        return tf.reduce_mean(coef)

    # end of MY Metrics----------------------------------------------------------------------


    def dice_coef_loss(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Sorenson (Soft) Dice loss
        Using -log(Dice) as the loss since it is better behaved.
        Also, the log allows avoidance of the division which
        can help prevent underflow when the numbers are very small.
        """
        intersection = tf.reduce_sum(prediction * target, axis=axis)
        p = tf.reduce_sum(prediction, axis=axis)
        t = tf.reduce_sum(target, axis=axis)
        numerator = tf.reduce_mean(intersection + smooth)
        denominator = tf.reduce_mean(t + p + smooth)
        dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

        return dice_loss

    def combined_dice_ce_loss(self, target, prediction, axis=(1, 2), smooth=0.0001):
        """
        Combined Dice and Binary Cross Entropy Loss
        """
        

        return self.weight_dice_loss*self.dice_coef_loss(target, prediction, axis, smooth) + \
            (1-self.weight_dice_loss)*K.losses.binary_crossentropy(target, prediction)

    def unet_model(self, imgs_shape, msks_shape,
                   dropout=0.2,
                   final=False):
        """
        U-Net Model
        ===========
        Based on https://arxiv.org/abs/1505.04597
        The default uses UpSampling2D (nearest neighbors interpolation) in
        the decoder path. The alternative is to use Transposed
        Convolution.
        """

        if not final:
            if self.use_upsampling:
                print("Using UpSampling2D")
            else:
                print("Using Transposed Convolution")

        num_chan_in = imgs_shape[self.concat_axis]
        num_chan_out = msks_shape[self.concat_axis]

        # You can make the network work on variable input height and width
        # if you pass None as the height and width
#        if self.channels_first:
#            self.input_shape = [num_chan_in, None, None]
#        else:
#            self.input_shape = [None, None, num_chan_in]

        self.input_shape = imgs_shape

        self.num_input_channels = num_chan_in

        inputs = K.layers.Input(self.input_shape, name="MRImages")

        # Convolution parameters
        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                            padding="same")

        encodeA = K.layers.Conv2D(
            name="encodeAa", filters=self.fms, **params)(inputs)
        encodeA = K.layers.Conv2D(
            name="encodeAb", filters=self.fms, **params)(encodeA)
        poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = K.layers.Conv2D(
            name="encodeBa", filters=self.fms*2, **params)(poolA)
        encodeB = K.layers.Conv2D(
            name="encodeBb", filters=self.fms*2, **params)(encodeB)
        poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = K.layers.Conv2D(
            name="encodeCa", filters=self.fms*4, **params)(poolB)
        if self.use_dropout:
            encodeC = K.layers.SpatialDropout2D(dropout)(encodeC)
        encodeC = K.layers.Conv2D(
            name="encodeCb", filters=self.fms*4, **params)(encodeC)

        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

        encodeD = K.layers.Conv2D(
            name="encodeDa", filters=self.fms*8, **params)(poolC)
        if self.use_dropout:
            encodeD = K.layers.SpatialDropout2D(dropout)(encodeD)
        encodeD = K.layers.Conv2D(
            name="encodeDb", filters=self.fms*8, **params)(encodeD)

        poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = K.layers.Conv2D(
            name="encodeEa", filters=self.fms*16, **params)(poolD)
        encodeE = K.layers.Conv2D(
            name="encodeEb", filters=self.fms*16, **params)(encodeE)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upE", size=(2, 2))(encodeE)
        else:
            up = K.layers.Conv2DTranspose(name="transconvE", filters=self.fms*8,
                                          **params_trans)(encodeE)
        concatD = K.layers.concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD")

        decodeC = K.layers.Conv2D(
            name="decodeCa", filters=self.fms*8, **params)(concatD)
        decodeC = K.layers.Conv2D(
            name="decodeCb", filters=self.fms*8, **params)(decodeC)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upC", size=(2, 2))(decodeC)
        else:
            up = K.layers.Conv2DTranspose(name="transconvC", filters=self.fms*4,
                                          **params_trans)(decodeC)
        concatC = K.layers.concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = K.layers.Conv2D(
            name="decodeBa", filters=self.fms*4, **params)(concatC)
        decodeB = K.layers.Conv2D(
            name="decodeBb", filters=self.fms*4, **params)(decodeB)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upB", size=(2, 2))(decodeB)
        else:
            up = K.layers.Conv2DTranspose(name="transconvB", filters=self.fms*2,
                                          **params_trans)(decodeB)
        concatB = K.layers.concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = K.layers.Conv2D(
            name="decodeAa", filters=self.fms*2, **params)(concatB)
        decodeA = K.layers.Conv2D(
            name="decodeAb", filters=self.fms*2, **params)(decodeA)

        if self.use_upsampling:
            up = K.layers.UpSampling2D(name="upA", size=(2, 2))(decodeA)
        else:
            up = K.layers.Conv2DTranspose(name="transconvA", filters=self.fms,
                                          **params_trans)(decodeA)
        concatA = K.layers.concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        convOut = K.layers.Conv2D(
            name="convOuta", filters=self.fms, **params)(concatA)
        convOut = K.layers.Conv2D(
            name="convOutb", filters=self.fms, **params)(convOut)

        prediction = K.layers.Conv2D(name="PredictionMask",
                                     filters=num_chan_out, kernel_size=(1, 1),
                                     activation="sigmoid")(convOut)

        model = K.models.Model(inputs=[inputs], outputs=[
                               prediction], name="2DUNet_Brats_Decathlon")

        optimizer = self.optimizer

        if final:
            model.trainable = False
        else:

            model.compile(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            if self.print_model:
                model.summary()

        return model


    def create_model(self, imgs_shape, msks_shape,
                     dropout=0.2,
                     final=False):
        """
        If you have other models, you can try them here
        """
        return self.unet_model(imgs_shape, msks_shape,
                               dropout=dropout,
                               final=final)

    def load_model(self, model_filename):
        """
        Load a model from Keras file
        """

        return K.models.load_model(model_filename, custom_objects=self.custom_objects)

    def print_openvino_mo_command(self, model_filename, input_shape):
        """
        Prints the command for the OpenVINO model optimizer step
        """
        model = self.load_model(model_filename)

        print("Convert the TensorFlow model to OpenVINO by running:\n")
        print("source /opt/intel/openvino_2021/bin/setupvars.sh")
        print("python $INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo_tf.py \\")
        print("       --saved_model_dir {} \\".format(model_filename))

        shape_string = "[1"
        for idx in range(len(input_shape)):
            shape_string += ",{}".format(input_shape[idx])
        shape_string += "]"

        print("       --input_shape {} \\".format(shape_string))
        print("       --model_name {} \\".format(self.inference_filename))
        print("       --output_dir {} \\".format(os.path.join(self.output_path, "FP32")))
        print("       --data_type FP32\n\n")



#
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,trainloader, valloader, num_train_samples, num_val_samples, model_input_shape, model_output_shape)->None:
    # def __init__(self, num_train_samples, num_val_samples, model_input_shape, model_output_shape)->None:

        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        #Instatiate the model that will be trained
        #self.model = get_model(model_input_shape,model_output_shape)
        self.model = get_model([128, 128, 1],[128, 128, 1])
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.model_input_shape = [128, 128, 1]
        self.model_output_shape = [128, 128, 1]
    
    def get_parameters(self, config):
        param = self.model.get_weights()
        return param

    def fit(self, parameters, config):
        #parameters is a list of numpy arrays representing the weights of the global model
        # Copy parameters sent by the server into client's local model
        self.model.set_weights(parameters) #9:40 in video
        epochs = config['local_epochs']
        history = self.model.fit(self.trainloader, epochs=epochs, validation_data=self.valloader, verbose=2)#, callbacks=model_callbacks)

        # Return the metrics along with the model weights
        results = {
            'loss': history.history['loss'][0],
            'dice_coef': history.history['dice_coef'][0],
            'soft_dice_coef': history.history['soft_dice_coef'][0],
            'iou_coef': history.history['iou_coef'][0],
            'precision': history.history['precision'][0],
            'accuracy': history.history['accuracy'][0],
            'specificity': history.history['specificity'][0]
            # 'val_loss': history.history['val_loss'][0],
            # 'val_dice_coef': history.history['val_dice_coef'][0],
            # 'val_soft_dice_coef': history.history['val_soft_dice_coef'][0]
        }
        return self.model.get_weights(), self.num_train_samples, results # for sending anything (like run time or metrics) to server

    def evaluate(self, parameters, config):
        # get global model to be evaluated on client's validation data
        self.model.set_weights(parameters)
        'check model.py line 76 ,81. Here we might need to add loss to the metrics so that it gets returned here> dont think so, loss is a normal return'
        loss, dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity = self.model.evaluate(self.valloader, verbose=2)
        return float(loss), self.num_val_samples, {"dice_coef": dice_coef,"soft_dice_coef": soft_dice_coef,"iou_coef":iou_coef,"precision":precision,"accuracy":accuracy,"specificity":specificity}


def get_client_fn(trainloaders, valloaders, num_train_samples_clients, num_val_samples_clients, model_input_shape, model_output_shape):

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
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_train_samples=num_train_samples_clients[int(cid)],# this indexes into the list and pulls out the number of samples this client model trains on
                            num_val_samples=num_val_samples_clients[int(cid)],
                            model_input_shape=model_input_shape,
                            model_output_shape=model_output_shape
                            )
        # return FlowerClient(
        #                     num_train_samples=num_train_samples_clients[int(cid)],# this indexes into the list and pulls out the number of samples this client model trains on
        #                     num_val_samples=num_val_samples_clients[int(cid)],
        #                     model_input_shape=model_input_shape,
        #                     model_output_shape=model_output_shape
        #                     )
    return client_fn

def get_model(input_shape, output_shape):
   u_net = unet() 
   model= u_net.create_model([128, 128, 1], [128, 128, 1], final=False)
   return model

# This method can go in server.py
def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        #if server_round > 50:
        #    lr = config.lr /10
        return {'lr':config.lr, 'local_epochs':config.local_epochs}
    
    return fit_config_fn

# This method can go in server.py    
def get_evaluate_fn(valset,input_shape,output_shape):
#def get_evaluate_fn(input_shape,output_shape):


    """Return an evaluation function for server-side (i.e. centralised) evaluation."""


    # The `evaluate` function will be called after every round by the strategy
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        print("evaluate function called after every round by strategy")
        model = get_model([128, 128, 1],[128, 128, 1])  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity = model.evaluate(valset, verbose=0)
        
        return loss, {"dice_coef": dice_coef,"soft_dice_coef": soft_dice_coef,"iou_coef":iou_coef,"precision":precision,"accuracy":accuracy,"specificity":specificity}
    return evaluate


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Aggregation function for (federated) evaluation metrics.

    It will aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply each metric of every client by the number of examples used
    dice_coef = [num_examples * m["dice_coef"] for num_examples, m in metrics]
    soft_dice_coef = [num_examples * m["soft_dice_coef"] for num_examples, m in metrics]
    iou_coef = [num_examples * m["iou_coef"] for num_examples, m in metrics]
    precision = [num_examples * m["precision"] for num_examples, m in metrics]
    accuracy = [num_examples * m["accuracy"] for num_examples, m in metrics]
    specificity = [num_examples * m["specificity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]


    # Aggregate and return custom metrics (weighted average)
    return {
        "dice_coef": sum(dice_coef) / sum(examples),
        "soft_dice_coef": sum(soft_dice_coef) / sum(examples),
        "iou_coef": sum(iou_coef) / sum(examples),#iou_coef
        "precision": sum(precision) / sum(examples),
        "accuracy": sum(accuracy) / sum(examples),
        "specificity": sum(specificity) / sum(examples)
    }

class CustomFedAvg(fl.server.strategy.FedAvg):#FedAdam
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_metrics = defaultdict(list)  # Store metrics for each client

    def aggregate_fit(self, rnd, results, failures):
        print("Aggregating fit")
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid  # Assuming the ClientProxy has a 'cid' attribute for client ID
            self.client_metrics[client_id].append(fit_res.metrics)
        aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        return aggregated_metrics
    #  Each key in the dictionary is a client ID, and the associated value is a list of metrics for that client over the rounds.

class MyServer(FlowerServer):
    """Flower server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_val_loss = float('inf')  # For early stopping
        self.rounds_without_improvement = 0  # For early stopping
        self.patience = 60  # Patience for early stopping
        self.global_parameters = None
        self.best_parameters = None
        self.window_size=10
        self.prev_moving_avg=float('inf')

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        recent_losses=[]
        """Run federated averaging for a number of rounds."""
        history = History()
        early_stopping_triggered = False

        # Initialize parameters
        log(INFO, "Initializing global parameters :)")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            print("in MyServer, about to evaluate")
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                print("Aggregated model: loss_centralized = ",loss_cen)
                recent_losses.append(loss_cen)
                if len(recent_losses) > self.window_size:  # Assuming you've set a window_size attribute for your class
                    recent_losses.pop(0)
                current_moving_avg = np.mean(recent_losses)
                if current_moving_avg >= self.prev_moving_avg:
                    # model got worse based on moving average
                    self.rounds_without_improvement += 1
                else:
                    # model got better based on moving average
                    self.best_parameters = deepcopy(self.parameters)
                    self.rounds_without_improvement = 0
                    self.prev_moving_avg = current_moving_avg


                if self.rounds_without_improvement >= self.patience:
                    print("Early stopping triggered due to no improvement in validation loss for", self.patience, "rounds.")
                    print("The final validation loss was: ", self.prev_val_loss, " which was achieved on round ", current_round - self.patience)
                    params = parameters_to_ndarrays(self.parameters)
                    self._save_model(params)
                    params = parameters_to_ndarrays(self.best_parameters)
                    self.save_modelbest(params)
                    self.global_parameters = deepcopy(params)
                    early_stopping_triggered = True
                    break

                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        if not early_stopping_triggered:
            params = parameters_to_ndarrays(self.parameters)
            self._save_model(params)
            self.global_parameters = deepcopy(params)
            print("Ended without early stopping")

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


    def _save_model(self,weights):
        model = get_model([128, 128, 1], [128, 128, 1])
        model.set_weights(weights)
        model.save(f'FLFedAvg_{cfg.num_clients}_clients_end_global_model.keras')
        print("Saved model to disk.")
    def _save_modelbest(self,weights):
        model = get_model([128, 128, 1], [128, 128, 1])
        model.set_weights(weights)
        model.save(f'FLFedAvg_{cfg.num_clients}_clients_Best_global_model.keras')
        print("Saved model to disk.")

def main(cfg: DictConfig) -> None:

    #generate random letter
    import random

    random_letter = chr(random.randint(65, 90))  # generate a random uppercase letter
    print("This run is letter ",random_letter)
    
    
    enable_tf_gpu_growth()
    # Parse input arguments
    print("Number of Clients: ",cfg.num_clients)
    print("Number of Clients per round: ",cfg.num_clients_per_round_fit)
    print("Number of Rounds: ",cfg.num_rounds)
    print("Number of Local Epochs: ",cfg.config_fit.local_epochs)
    print("Batch Size: ",cfg.batch_size)
    
    # 1. Load Data
    print("Loading and Partitioning Data")
    trainloaders, valloaders, valloader_global, testloader, input_shape, output_shape = load_datasets(cfg.num_clients, cfg.batch_size)
    print("Valloader len: ",len(valloader_global))
    print("Testloader len: ",len(testloader))

    
    # Check that the batches
    print("Data Loaded")
    num_clients=len(trainloaders)
    if cfg.num_clients != num_clients:
        print("Error: dataloader did not return the correct number of training sets: ",num_clients,"!=",cfg.num_clients)
    train_sample_dict={
        2:[20305,20305],#updated
        4:[10230,10230,10075,10075],
        8:[5270,5270,5115,5115,5115,5115,5115,5115] 
    }
    val_sample_dict={
        2:[2170,2170],#updated
        4:[1085,1085,1085,1085],
        8:[465,465,465,465,465,465,465,465] 
    }
    # get num training samples for each client from train_sample_dict
    num_train_samples_clients = train_sample_dict[cfg.num_clients]
    # get num vallidation sample for each client from val_sample_dict
    num_val_samples_clients = val_sample_dict[cfg.num_clients]
    print("fraction_evaluate = 0 ---------------------------------------------------------------------")
    # Create FedAvg strategy
    MyStrategy = CustomFedAvg(
        fraction_fit=1,  # Sample 10% of available clients for training
        fraction_evaluate=1,  # Sample 5% of available clients for evaluation
        #setting to 0 gave shape compatibility error
        min_fit_clients=cfg.num_clients_per_round_fit ,  # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.num_clients_per_round_eval ,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            cfg.num_clients * 1
        ),  # Wait until at least n clients are available
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(valloader_global, input_shape, output_shape),  # global evaluation function # evaluate aggregated model on global val set
    ) # sending in valloader_global to evaluate aggreagated model
    # get_evaluate_fn gives the server an emaulation function. The server then calls this function after each round to evaluate the global model on the global validation set
    
    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    # NOTE commenting out resouces since docs say it is auto determined by ray
    client_resources = {
        "num_cpus": cfg.num_cpus,
        "num_gpus": cfg.num_gpus,
    }
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(trainloaders, valloaders,num_train_samples_clients, num_val_samples_clients, input_shape,output_shape),
        num_clients=cfg.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        server=MyServer(client_manager=SimpleClientManager(),strategy=MyStrategy),
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        }
    ) 
    print("Got to end of simulation")
    # Initialize a new figure for plotting
    try:
        
        # Loop over each client's metrics for plotting
        for idx, (client_id, metrics_list) in enumerate(MyStrategy.client_metrics.items()):
            plt.figure(figsize=(15, 5))
            losses = [metrics['loss'] for metrics in metrics_list]
            dice_coefs = [metrics['dice_coef'] for metrics in metrics_list]
            soft_dice_coefs = [metrics['soft_dice_coef'] for metrics in metrics_list]
            # save as npy file
            np.save(f'{random_letter}_FLFedAvg_{cfg.num_clients}_client_{client_id}_losses.npy', losses)
            np.save(f'{random_letter}_FLFedAvg_{cfg.num_clients}_client_{client_id}_dice_coefs.npy', dice_coefs)
            np.save(f'{random_letter}_FLFedAvg_{cfg.num_clients}_client_{client_id}_soft_dice_coefs.npy', soft_dice_coefs)

            losses = [100.0 * data for data in losses]
            dice_coefs = [100.0 * data for data in dice_coefs]
            soft_dice_coefs = [100.0 * data for data in soft_dice_coefs]

            # Plotting Loss for the client
            plt.subplot(len(MyStrategy.client_metrics), 3, idx * 3 + 1)
            plt.plot(losses, label=f'Client {client_id} Loss')
            plt.title(f'Client {client_id} Loss over Rounds')
            plt.xlabel('Rounds')
            plt.ylabel('Loss')
            plt.xticks(range(0,len(losses)+10,10))
            # plt.ylim(0, 100)  # set y-axis range to 0-100
            plt.legend()

            # Plotting Dice Coefficient for the client
            plt.subplot(len(MyStrategy.client_metrics), 3, idx * 3 + 2)
            plt.plot(dice_coefs, label=f'Client {client_id} Dice Coefficient')
            plt.title(f'Client {client_id} Dice Coefficient over Rounds')
            plt.xlabel('Rounds')
            plt.ylabel('Dice Coefficient (%)')
            plt.xticks(range(0,len(dice_coefs)+10,10))
            # plt.ylim(0, 100)  # set y-axis range to 0-100
            plt.legend()

            # Plotting Soft Dice Coefficient for the client
            plt.subplot(len(MyStrategy.client_metrics), 3, idx * 3 + 3)
            plt.plot(soft_dice_coefs, label=f'Client {client_id} Soft Dice Coefficient')
            plt.title(f'Client {client_id} Soft Dice Coefficient over Rounds')
            plt.xlabel('Rounds')
            plt.ylabel('Soft Dice Coefficient (%)')
            plt.xticks(range(0,len(soft_dice_coefs)+10,10))
            # plt.ylim(0, 100)  # set y-axis range to 0-100
            plt.legend()

            # Adjust layout and display the plots
            plt.tight_layout()
            plt.savefig(f'{random_letter}_FLFedAvg_{cfg.num_clients}_clients_plots_FedAvg_client_{client_id}.png')
            plt.figure()
    except:
        print("Failed to plot client metrics")

    try:
        for idx, (client_id, metrics_list) in enumerate(MyStrategy.client_metrics.items()):
            plt.figure(figsize=(15, 5))
            iou = [metrics['iou_coef'] for metrics in metrics_list]
            # save as npy file
            np.save(f'{random_letter}_FLFedAvg_{cfg.num_clients}_client_{client_id}_iou.npy', iou)

            iou = [100.0 * data for data in iou]

            # Plotting Loss for the client
            plt.subplot(len(MyStrategy.client_metrics), 3, idx * 3 + 1)
            plt.plot(iou, label=f'Client {client_id} IOU')
            plt.title(f'Client {client_id} IOU over Rounds')
            plt.xlabel('Rounds')
            plt.ylabel('IOU')
            plt.xticks(range(0,len(iou)+10,10))
            # plt.ylim(0, 100)  # set y-axis range to 0-100
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{random_letter}_FLFedAvg_{cfg.num_clients}_clients_iou_plots_FedAvg_client_{client_id}.png')
            plt.figure()
    except:
        print("couldn't plot iou")
    #try catch the below plotting code  
    try:
        print(f"{history.metrics_centralized = }")

        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["dice_coef"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("Dice Coefficient (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's Dice Coefficient with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_DC_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
        np.save(f'{random_letter}_global_model_DC_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["dice_coef"])
    except: 
        print("Failed to plot global model dice coefficient and save raw data")

    try:
        # print(f"{history.metrics_centralized = }")
        np.save(f'{random_letter}_global_model_IOU_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["iou_coef"])
        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["iou_coef"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("IOU (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's IOU with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_IOU_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
        
    except: 
        print("Failed to plot global model dice coefficient and save raw data")
    
    try:
        # print(f"{history.metrics_centralized = }")

        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["soft_dice_coef"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("Soft Dice Coefficient (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's Soft Dice Coefficient with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_soft_DC_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
        np.save(f'{random_letter}_global_model_softDC_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["soft_dice_coef"])
    except: 
        print("Failed to plot global model soft dice coefficient and save raw data")
        np.save(f'{random_letter}_global_model_softDC_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["soft_dice_coef"])

    try:
        # print(f"{history.metrics_centralized = }")
        np.save(f'{random_letter}_global_model_Precision_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["precision"])

        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["precision"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("Precision (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's Precision with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_Precision_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
    except: 
        print("Failed to plot global model precision and save raw data")

    try:
        # print(f"{history.metrics_centralized = }")
        np.save(f'{random_letter}_global_model_Accuracy_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["accuracy"])

        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["accuracy"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's accuracy with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_accuracy_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
    except: 
        print("Failed to plot global model Accuracy and save raw data")

    try:
        # print(f"{history.metrics_centralized = }")
        np.save(f'{random_letter}_global_model_Specificity_{cfg.num_clients}_clients_FedAvg.npy', history.metrics_centralized["specificity"])

        #save data for this plot with np.save
        global_dice_centralised = history.metrics_centralized["specificity"]
        round = [data[0] for data in global_dice_centralised]
        dice = [100.0 * data[1] for data in global_dice_centralised]
        plt.plot(round, dice)
        plt.grid()
        plt.ylabel("Specificity (%)")
        plt.xlabel("Round")
        plt.title(f"Aggregated model's specificity with {cfg.num_clients} clients")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_specificity_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
    except: 
        print("Failed to plot global model specificity and save raw data")

    #plot history.losses_centralized
    try:
        global_loss_centralised = history.losses_centralized
        round = [data[0] for data in global_loss_centralised]
        loss = [data[1] for data in global_loss_centralised]
        plt.plot(round, loss)
        plt.grid()
        plt.ylabel("Loss (%)")
        plt.xlabel("Round")
        plt.title(f"BRATS - IID - 2 clients with {cfg.num_clients} clients per round")
        # save the plot
        plt.savefig(f'{random_letter}_global_model_loss_{cfg.num_clients}_clients_FedAvg.png')
        plt.figure()
        # save the raw data
        np.save(f'{random_letter}_global_model_loss_{cfg.num_clients}_clients_FedAvg.npy', history.losses_centralized)
    except:
        print("Failed to plot global model loss")

    # Get global model and evaluate using test set
    # only load model if file exists
    try:
        if os.path.isfile(f'FLFedAvg_{cfg.num_clients}_clients_end_global_model.keras'):
            print("Loading end model from file")
            model = get_model([128, 128, 1], [128, 128, 1])
            model.load_weights(f'FLFedAvg_{cfg.num_clients}_clients_end_global_model.keras')
            loss, dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity = model.evaluate(testloader, verbose=1)
            print()
            print("-----TEST RESULTS-----")
            print("Test Loss: ",loss)
            print("Test Dice Coefficient: ",dice_coef)
            print("Test Soft Dice Coefficient: ",soft_dice_coef)
            print("Test IOU Coefficient: ",iou_coef)
            print("Test Precision: ",precision)
            print("Test Accuracy: ",accuracy)
            print("Test Specificity: ",specificity)
    except:
        print("Failed to load end model from file and evaluate")
    try:
        if os.path.isfile(f'FLFedAvg_{cfg.num_clients}_clients_Best_global_model.keras'):
            print("Loading best model from file")
            model = get_model([128, 128, 1], [128, 128, 1])
            model.load_weights(f'FLFedAvg_{cfg.num_clients}_clients_Best_global_model.keras')
            loss, dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity = model.evaluate(testloader, verbose=1)
            print()
            print("-----TEST RESULTS-----")
            print("Test Loss: ",loss)
            print("Test Dice Coefficient: ",dice_coef)
            print("Test Soft Dice Coefficient: ",soft_dice_coef)
            print("Test IOU Coefficient: ",iou_coef)
            print("Test Precision: ",precision)
            print("Test Accuracy: ",accuracy)
            print("Test Specificity: ",specificity)
    except:
        print("Failed to load best model from file and evaluate")
    #try catch the below block of code
    
    #do the same but with stategy.global_parameters
    try:
        print("Loading model from global_parameters")
        model = get_model([128, 128, 1], [128, 128, 1])
        model.set_weights(MyStrategy.global_parameters)
        loss, dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity = model.evaluate(testloader, verbose=2)
        print()
        print("-----TEST RESULTS-----")
        print("Test Loss: ",loss)
        print("Test Dice Coefficient: ",dice_coef)
        print("Test Soft Dice Coefficient: ",soft_dice_coef)
        print("BOTH TESTS SHOULD BE THE SAME")
    except:
        print("Failed to load model from global_parameters")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change config_name from terminal.")
    parser.add_argument('--config_name', default='base', help='Name of the config to be used')
    args = parser.parse_args()

    config_path = os.path.abspath("conf")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=args.config_name)
        main(cfg)

