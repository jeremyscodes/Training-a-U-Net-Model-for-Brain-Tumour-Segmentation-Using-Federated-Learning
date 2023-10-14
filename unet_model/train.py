#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
This module loads the data from data.py, creates a TensorFlow/Keras model
from model.py, trains the model on the data, and then saves the
best model.
"""

import datetime
import os

import tensorflow as tf  # conda install -c anaconda tensorflow
import settings   # Use the custom settings.py file for default parameters

from dataloader import DatasetGenerator, get_decathlon_filelist

import numpy as np

from argparser import args

import warnings
warnings.filterwarnings("ignore")

"""
For best CPU speed set the number of intra and inter threads
to take advantage of multi-core systems.
See https://github.com/intel/mkl-dnn
"""

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# If hyperthreading is NOT enabled, then use
#os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

os.environ["KMP_BLOCKTIME"] = str(args.blocktime)

os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["INTRA_THREADS"] = str(args.num_threads)
os.environ["INTER_THREADS"] = str(args.num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf

    print("We are using Tensorflow version {}".format(tf.__version__))

import matplotlib.pyplot as plt

def save_training_plots(history, output_path):
    # Define a function to save plots
    def save_plot(values, val_values, ylabel, title, filename):
        plt.figure(figsize=(6, 4))
        plt.plot(values)
        plt.plot(val_values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(filename)
        plt.close()
    
    # Save Loss Plot
    save_plot(history.history['loss'], history.history['val_loss'], 'Loss', 'Model Loss', 
              os.path.join(output_path, 'loss_plot.png'))
    
    # Save Dice Coefficient Plot
    save_plot(history.history['dice_coef'], history.history['val_dice_coef'], 'Dice Coefficient', 
              'Model Dice Coefficient', os.path.join(output_path, 'dice_coef_plot.png'))
    
    # Save Soft Dice Coefficient Plot
    save_plot(history.history['soft_dice_coef'], history.history['val_soft_dice_coef'], 
              'Soft Dice Coefficient', 'Model Soft Dice Coefficient', 
              os.path.join(output_path, 'soft_dice_coef_plot.png'))

# You can now call this function to save the plots:
# save_training_plots(history, 'path_to_save_plots')


# This function can be used to plot the training history like this:
# plot_training_history(history)


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))
    print("Runtime arguments = {}".format(args))
    test_intel_tensorflow() # Print tensorflow version
    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Medical Decathlon directory to a TensorFlow data loader ...")
    print("-" * 30)

    trainFiles, validateFiles, testFiles = get_decathlon_filelist(data_path=args.data_path, seed=args.seed, split=args.split)
    print("Number of trainFiles")
    print(len(trainFiles))
    ds_train = DatasetGenerator(trainFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=True, seed=args.seed)
    print(len(ds_train))
    ds_validation = DatasetGenerator(validateFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)
    ds_test = DatasetGenerator(testFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """
    if args.use_pconv:
        from model_pconv import unet
    else:
        from model import unet

    unet_model = unet(channels_first=args.channels_first,
                 fms=args.featuremaps,
                 output_path=args.output_path,
                 inference_filename=args.inference_filename,
                 learning_rate=args.learningrate,
                 weight_dice_loss=args.weight_dice_loss,
                 use_upsampling=args.use_upsampling,
                 use_dropout=args.use_dropout,
                 print_model=args.print_model)
   

    model = unet_model.create_model(
        ds_train.get_input_shape(), ds_train.get_output_shape())
    print("ds_train.get_input_shape(), ds_train.get_output_shape()")
    print(
        ds_train.get_input_shape(), ds_train.get_output_shape())

    print("Get callbacks")
    model_filename, model_callbacks = unet_model.get_callbacks() 
    

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    history = model.fit(ds_train,
              epochs=args.epochs,
              validation_data=ds_validation,
              verbose=2,
              callbacks=model_callbacks)
    
    # plot the training loss and metrics
    save_training_plots(history, './')


    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    model_filename = './output/2d_unet_decathlon/'

    unet_model.evaluate_model(model_filename, ds_test)

    """
    Step 5: Print the command to convert TensorFlow model into OpenVINO format with model optimizer.
    """
    print("-" * 30)
    print("-" * 30)
    unet_model.print_openvino_mo_command(
        model_filename, ds_test.get_input_shape())

    print(
        "Total time elapsed for program = {} seconds".format(
            datetime.datetime.now() -
            START_TIME))
    print("Stopped script on {}".format(datetime.datetime.now()))

    
