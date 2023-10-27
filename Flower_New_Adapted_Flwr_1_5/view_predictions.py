import os
from keras.models import load_model
import numpy as np

from ang_loader import load_datasets
from my_sim import unet

import tensorflow as tf
from tensorflow import keras as K

trainloaders, valloaders, valloader_global, testloader, input_shape, output_shape = load_datasets(num_partitions=2,batch_size=20,  val_ratio=0.1)

@tf.function
@K.saving.register_keras_serializable()
def dice_coef( target, prediction, axis=(1, 2), smooth=0.0001):
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

@tf.function
@K.saving.register_keras_serializable()
def iou_coef( target, prediction, axis=(1, 2), smooth=0.0001):
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

@tf.function
@K.saving.register_keras_serializable()
def soft_dice_coef( target, prediction, axis=(1, 2), smooth=0.0001):
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

@tf.function
@K.saving.register_keras_serializable()
def specificity( y_true, y_pred):
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

@tf.function
@K.saving.register_keras_serializable()
def precision( target, prediction, axis=(1, 2), smooth=0.0001):
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

@tf.function
@K.saving.register_keras_serializable()
def accuracy( target, prediction, axis=(1, 2), smooth=0.0001):
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

@tf.function
@K.saving.register_keras_serializable()
def dice_coef_loss( target, prediction, axis=(1, 2), smooth=0.0001):
    print("Target shape", target.shape)
    print("Prediction shape", prediction.shape)
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


# Define custom objects
custom_objects = {
    "dice_coef_loss": dice_coef_loss,
    "dice_coef": dice_coef,
    "soft_dice_coef": soft_dice_coef,
    "iou_coef": iou_coef,
    "precision": precision,
    "accuracy": accuracy,
    "specificity": specificity
}


import matplotlib.pyplot as plt
import numpy as np

# Define the Dice score function
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# List of model paths
model_paths = []  # Add your model paths here
# model_paths.append("central_model.pb")
'''
Traceback (most recent call last):
  File "/home-mscluster/jstott/Research_Code/Flower_New_Adapted_Flwr_1_5/view_predictions.py", line 170, in <module>
    models = [load_model(path, custom_objects=custom_objects, compile=False) for path in model_paths]
  File "/home-mscluster/jstott/Research_Code/Flower_New_Adapted_Flwr_1_5/view_predictions.py", line 170, in <listcomp>
    models = [load_model(path, custom_objects=custom_objects, compile=False) for path in model_paths]
  File "/home-mscluster/jstott/anaconda3/envs/flwr_new_tf/lib/python3.9/site-packages/keras/src/saving/saving_api.py", line 262, in load_model
    return legacy_sm_saving_lib.load_model(
  File "/home-mscluster/jstott/anaconda3/envs/flwr_new_tf/lib/python3.9/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home-mscluster/jstott/anaconda3/envs/flwr_new_tf/lib/python3.9/site-packages/h5py/_hl/files.py", line 567, in __init__
    fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
  File "/home-mscluster/jstott/anaconda3/envs/flwr_new_tf/lib/python3.9/site-packages/h5py/_hl/files.py", line 231, in make_fid
    fid = h5f.open(name, flags, fapl=fapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 106, in h5py.h5f.open
OSError: Unable to open file (file signature not found)

'''
nums = [2,4,8]
for i,num in enumerate(nums):
    model_path = f'FLFedAvg_{num}_clients_Best_global_model.keras'
    model_paths.append(model_path)
# Load models   
models = [load_model(path, custom_objects=custom_objects, compile=False) for path in model_paths]

# Select 5 slices from different batches

# slices = []
# for i, batch in enumerate(testloader):
#     inputs, targets = batch
#     slices.append((inputs[0], targets[0]))  # taking the first slice from each batch
#     if i == num_samples - 1:
#         break
num_batches = len(testloader)
print("Num batches ",num_batches)
num_slices_per_scan = testloader.num_slices_per_scan
print("Num slices per scan = ",num_slices_per_scan)

print("haha")
num_samples = 5
suitable_samples=0
# Plot
fig, axes = plt.subplots(nrows=num_samples, ncols=len(models) + 2, figsize=(15, 15))

for i, batch in enumerate(testloader):
    if suitable_samples >= num_samples:
        break  # Stop if you've processed enough batches
    print("Batch ",i)
    print("length of batch ",len(batch))
    input_slices, targets = batch
    print("length of input slices ",len(input_slices))
    print("length of targets ",len(targets))
    count=0
    while count < len(input_slices): 
        
        input_slice = input_slices[count]
        target = targets[count]

        if not np.any(target):
            print("empty, skip")
            count+=1  
            continue

        # check if dice score is very high or very low based on worst model 
        print("Sample from batch ", i, " Slice ", count)
        prediction = models[0].predict(input_slice[np.newaxis, ...],verbose=0)#.squeeze()
        dice = dice_coef(target, prediction.round()).numpy()
        print("Dice score = ",dice)
        if dice>0.8:            
            print("using sample")
            # MRI slice
            axes[suitable_samples, 0].imshow(input_slice.squeeze(), cmap='gray')
            axes[suitable_samples, 0].set_title('Original Image')
            # Ground truth
            if np.any(target):
                axes[suitable_samples, 1].imshow(input_slice.squeeze(), cmap='gray', alpha=0.5)
                axes[suitable_samples, 1].imshow(target.squeeze(), cmap='Reds', alpha=0.5)
                axes[suitable_samples, 1].set_title('Ground Truth')
            else:
                axes[suitable_samples, 1].imshow(input_slice.squeeze(), cmap='gray')
                axes[suitable_samples, 1].set_title('No Tumour Present')
            # Predictions
            for k, model in enumerate(models):
                prediction = model.predict(input_slice[np.newaxis, ...],verbose=0)#.squeeze()
                dice = dice_coef(target, prediction.round())
                axes[suitable_samples, k+2].imshow(input_slice.squeeze(), cmap='gray', alpha=0.5)
                axes[suitable_samples, k+2].imshow(prediction.squeeze(), cmap='Reds', alpha=0.5)
                axes[suitable_samples, k+2].set_title(f'Model {k+1} Dice: {dice:.3f}')
            suitable_samples+=1
            count+=50
            print("Number of samples added = ",suitable_samples)
          

print("Finished")
# Adjust and show
plt.tight_layout()
import os

output_dir = "Prediction_Results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, "results.png"))

