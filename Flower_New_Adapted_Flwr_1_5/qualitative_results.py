import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

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

def get_suitable_slices(testloader, model, num_samples=10, skip_slices=10):
    suitable_slices = []
    suitable_targets = []
    counter = 0
    batch_idx = 0
    
    while counter < num_samples and batch_idx < len(testloader):
        batch_data, batch_targets = testloader[batch_idx]
        print("batch_data shape", batch_data.shape)
        for slice_idx in range(batch_data.shape[0]):  # iterate over each slice in the batch
            input_slice = batch_data[slice_idx]
            target = batch_targets[slice_idx]
            print("input_slice shape", input_slice.shape, "indx=", slice_idx)
            
            prediction = model.predict(input_slice[np.newaxis, ...], verbose=0)
            dice = dice_coef(target[np.newaxis, ...], prediction.round()).numpy()
            
            if dice > 0.8 and np.any(target):  # Check dice score and ensure tumor exists
                suitable_slices.append(input_slice)
                suitable_targets.append(target)
                counter += 1
                if counter >= num_samples:
                    break

        batch_idx += 1

    return suitable_slices, suitable_targets



import matplotlib.pyplot as plt

def plot_suitable_slices(models, suitable_slices, suitable_targets):
    num_samples = len(suitable_slices)
    fig, axes = plt.subplots(num_samples, len(models) + 2, figsize=(15, 15))
    
    for i, (input_slice, target) in enumerate(zip(suitable_slices, suitable_targets)):
        # MRI slice
        axes[i, 0].imshow(input_slice.squeeze(), cmap='gray')
        axes[i, 0].set_title('Original Image')
        # Ground truth
        axes[i, 1].imshow(input_slice.squeeze(), cmap='gray', alpha=0.5)
        axes[i, 1].imshow(target.squeeze(), cmap='Reds', alpha=0.5)
        axes[i, 1].set_title('Ground Truth')
        # Predictions
        for k, model in enumerate(models):
            prediction = model.predict(input_slice[np.newaxis, ...], verbose=0)
            dice = dice_coef(target, prediction.round())
            axes[i, k+2].imshow(input_slice.squeeze(), cmap='gray', alpha=0.5)
            axes[i, k+2].imshow(prediction.squeeze(), cmap='Reds', alpha=0.5)
            axes[i, k+2].set_title(f'Model {k+1} Dice: {dice:.3f}')
    
    plt.tight_layout()
    plt.show()

# Assume models is a list containing all your models
from ang_loader import load_datasets
from keras.models import load_model

trainloaders, valloaders, valloader_global, testloader, input_shape, output_shape = load_datasets(num_partitions=2,batch_size=1,  val_ratio=0.1)
model_paths = []
nums = [2,4,8]
for i,num in enumerate(nums):
    model_path = f'FLFedAvg_{num}_clients_Best_global_model.keras'
    model_paths.append(model_path)
# Load models   
models = [load_model(path, custom_objects=custom_objects, compile=False) for path in model_paths]
suitable_slices, suitable_targets = get_suitable_slices(testloader, models[0])
if len(suitable_slices) == 0:
    print("No suitable slices found that meet the criteria.")
else:
    plot_suitable_slices(models, suitable_slices, suitable_targets)

