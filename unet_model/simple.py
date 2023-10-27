import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import unet
from dataloader import DatasetGenerator, get_decathlon_filelist
from tensorflow import keras as K
# Ensure GPU memory growth is limited to avoid out-of-memory issues
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

# Constants and paths
data_path = os.path.join('..', 'Task01_BrainTumour')
crop_dim = 128
batch_size = 20
seed = 816

# Get file lists
trainFiles, validateFiles, testFiles = get_decathlon_filelist(data_path=data_path, seed=seed)

# Generate testing dataset
ds_test = DatasetGenerator(testFiles, batch_size=batch_size, crop_dim=[crop_dim, crop_dim], augment=False, seed=seed)

# Load the model with custom objects
# central_model_path = os.path.join('..','unet_model', 'output', '2d_unet_decathlon')  # Adjust this path based on where your central model is saved
# central_model = load_model(central_model_path, custom_objects=unet().custom_objects)

fl_model = load_model('FL_Models/FLFedAvg_8_clients_Best_global_model.keras', custom_objects=custom_objects, compile=False)
print("FL models loaded")
optimizer = K.optimizers.Adam(learning_rate=0.0001)
metrics = [dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity]

# for model in fl_models:
fl_model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=metrics)
print("FL models compiled")

def calc_dice(target, prediction, smooth=0.0001):
    """
    Sorensen Dice coefficient using numpy for individual slices
    """
    prediction = np.round(prediction)

    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef

dice_scores = []

# Loop through all slices in the dataset. this is the same as .evaluate for the dice score
# for batch_images, batch_labels in ds_test:
#     for idx in range(batch_images.shape[0]):  # Loop over each slice in the batch
#         prediction_slice = fl_model.predict(batch_images[idx:idx+1])
#         dice_score = calc_dice(batch_labels[idx,:,:,0], prediction_slice[0,:,:,0])
        
#         if dice_score > 0.7:
#             print(f"Dice score for slice {idx}: {dice_score:.4f}")
        
#         dice_scores.append(dice_score)

# # Compute the average Dice score across all slices
# average_dice_score = np.mean(dice_scores)

# print(f"Average Dice Coefficient (per slice): {average_dice_score:.4f}")

# Evaluate the model
loss, dice, soft_dice, iou, prec, acc, spec = fl_model.evaluate(ds_test, batch_size=batch_size, verbose=1)  # Assuming dice coefficient is the only metric for simplicity

#print all metrics
print("Loss: ", loss)
print("Dice Coefficient: ", dice)
print("Soft Dice Coefficient: ", soft_dice)
print("IoU Coefficient: ", iou)
print("Precision: ", prec)
print("Accuracy: ", acc)
print("Specificity: ", spec)

print(metrics)
print("8 client model")

