print("This is the output I hope you see")
import os
from tensorflow.keras.models import load_model
from tensorflow import keras as K
from model import unet
import tensorflow as tf
print("Loaded tensorflow")
data_path = os.path.join('..', 'Task01_BrainTumour')
print(data_path)
#working to here so far

crop_dim=128  # Original resolution (240)
batch_size = 20
seed=816
train_test_split=0.60

from dataloader import DatasetGenerator, get_decathlon_filelist

trainFiles, validateFiles, testFiles = get_decathlon_filelist(data_path=data_path, seed=seed)
# # save trainFiles to a .txt
# file = open('testFiles.txt','w')
# for item in testFiles:
#     file.write(item+'\n')
# file.close()
# # These are the correct test files

# # TODO: Fill in the parameters for the dataset generator to return the `testing` data
ds_test = DatasetGenerator(testFiles, 
                           batch_size=batch_size, 
                           crop_dim=[crop_dim, crop_dim], 
                           augment=False, 
                           seed=seed)


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

def evaluate_models(ds, central_model_dir, fl_model_dir):
    print("method called")
    # Load the central model
    # central_model = load_model(central_model_dir, custom_objects=unet().custom_objects)

    # Load the FL models
    fl_model_names = [
        'FLFedAvg_2_clients_Best_global_model.keras',
        'FLFedAvg_4_clients_Best_global_model.keras',
        'FLFedAvg_8_clients_Best_global_model.keras'
    ]
    fl_models = [load_model(os.path.join(fl_model_dir, model_name), custom_objects=custom_objects, compile=False) for model_name in fl_model_names]
    
    # # Compile the FL models
    optimizer = K.optimizers.Adam(learning_rate=0.0001)
    metrics = [dice_coef, soft_dice_coef, iou_coef, precision, accuracy, specificity]
    print("Models loaded")
    for model in fl_models:
        model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=metrics)
    print("Models compiled")
    # all_models = [central_model] + fl_models
    names = ['Centralized Model','2 Client Aggregated Model', '4 Client Aggregated Model', '8 Client Aggregated Model']
    # loss, dice, soft_dice, iou, prec, acc, spec = central_model.evaluate(ds, batch_size=batch_size, verbose=1)
    # Iterate over each model and evaluate
    for model, name in zip(fl_models, names):
        print(f"Evaluating {name}...")
        loss, dice, soft_dice, iou, prec, acc, spec = model.evaluate(ds, batch_size=batch_size, verbose=0)
        print(f"Results for {name}:")
        print(f"Loss: {loss:.4f}")
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"Soft Dice Coefficient: {soft_dice:.4f}")
        print(f"IoU Coefficient: {iou:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Specificity: {spec:.4f}")
        print("-" * 40)

central_model_dir = os.path.join('..','unet_model', 'output', '2d_unet_decathlon') # Adjust this path based on where your central model is saved
fl_model_dir = './FL_Models'  # Directory containing the FL models

# # Call the function to evaluate the models
try:
    # existing evaluation code...
    evaluate_models(ds_test, central_model_dir, fl_model_dir)
except Exception as e:
    print(f"Error during evaluation: {e}")

print("ended")
