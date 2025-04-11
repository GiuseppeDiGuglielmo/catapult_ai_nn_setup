import catapult_ai_nn

import os
import shutil
import sys
import hls4ml
import yaml
import pickle
import argparse
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from qkeras import *
from sklearn.metrics import accuracy_score

# Usage:
#    python3 model.py 1
#      will build the model, perform analysis, generate HLS C++ and then synthesize it

## Function to create a simple Convolutional Neural Network model
import tensorflow as tf
def create_model():
    model = tf.keras.Sequential()
    model.add(QConv2DBatchnorm(filters=8, kernel_size=3, padding='same', strides=2, input_shape=(28,28,1), 
        kernel_quantizer=quantizers.quantized_bits(bits=8, integer=0), bias_quantizer=quantizers.quantized_bits(bits=8, integer=0)))
    #Compiling the model
    model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    return model

if __name__ == '__main__':

    print("============================================================================================")
    print("HLS4ML Version: ",hls4ml.__version__)

    parser = argparse.ArgumentParser(description='Configure and convert the model for Catapult HLS')
    parser.add_argument('--reuse_factor', type=int, default=1, help='Specify the ReuseFactor value')
    parser.add_argument('--synth', action='store_true', help='Specify whether to perform Catapult build and synthesis')

    args = parser.parse_args()

    # Define output files/location:
    model_name = 'simple'
    proj_name = 'myproject'
    out_dir = 'my-Catapult-test_asic' + str(args.reuse_factor)

    print("")
    print("Settings:")
    print('  model_name = '+model_name)
    print('  proj_name  = '+proj_name)
    print('  out_dir    = '+out_dir)
    print("")

    # Determine the directory containing this model.py script in order to locate the associated .dat file
    sfd = os.path.dirname(__file__)

    # Check if mnist_data.dat exists
    file_path = sfd + "/mnist_data.dat"

    if os.path.exists(file_path):
        print("Reading mnist data from file '" + file_path + "'")
        # Load the data from the existing .dat file
        with open(file_path, 'rb') as file:
            loaded_mnist_data = pickle.load(file)
    else:
        print("Downloading and converting mnist data...")
        # Assuming you have already loaded the MNIST data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Define a dictionary to store the data
        mnist_data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }

        # Save the data to the .dat file
        with open(file_path, 'wb') as file:
            pickle.dump(mnist_data, file)
        print("Mnist data saved to file '" + file_path + "'")

    # To load the data from the .dat file later:
    with open(file_path, 'rb') as file:
        loaded_mnist_data = pickle.load(file)

    # You can access the data as follows:
    x_train = loaded_mnist_data["x_train"]
    y_train = loaded_mnist_data["y_train"]
    x_test = loaded_mnist_data["x_test"]
    y_test = loaded_mnist_data["y_test"]

    # Normalize and reshape the dataset
    x_train = x_train[:10].astype('float32') / 255
    x_test = x_test[:10].astype('float32') / 255
    x_train = x_train[:10].reshape((10, 28, 28, 1))
    x_test = x_test[:10].reshape((10, 28, 28, 1))

    ## Create the model
    print("============================================================================================")
    print("Creating the model")
    model = create_model()
    model.summary()

    print("============================================================================================")
    print("Configuring HLS4ML")

    config_ccs = catapult_ai_nn.config_for_dataflow(model=model, x_test=x_test, y_test=y_test, num_samples=20, 
                                                   granularity='name', default_precision='ac_fixed<16,6,true>', 
                                                   default_reuse_factor=args.reuse_factor, max_precision='ac_fixed<16,6,true>',
                                                   project_name=proj_name, 
                                                   output_dir=out_dir, 
                                                   tech='asic',
                                                   asiclibs='saed32rvt_tt0p78v125c_beh',
                                                   asicfifo='hls4ml_lib.mgc_pipe_mem',
                                                   clock_period=10,
                                                   io_type='io_stream',
                                                   csim=1, SCVerify=1, Synth=1)

    print("============================================================================================")
    print("HLS4ML converting model to HLS C++")
    hls_model_ccs = catapult_ai_nn.generate_dataflow(model, config_ccs)

    print("============================================================================================")
    print("Compiling HLS C++ model")
    hls_model_ccs.compile()

    if args.synth:
        print("============================================================================================")
        print("Synthesizing HLS C++ model using Catapult")
        hls_model_ccs.build(csim=True, synth=True, cosim=False, validation=True, vsynth=False)
        # hls_model_ccs.build()
    else:
        print("============================================================================================")
        print("Skipping HLS")
        print("- To run Catapult directly from the shell:")
        print('    cd ' + out_dir + '; catapult -file build_prj.tcl')
        print("- To run directly in the current Catapult session:")
        print('    set_working_dir ' + out_dir)
        print('    dofile build_prj.tcl')
