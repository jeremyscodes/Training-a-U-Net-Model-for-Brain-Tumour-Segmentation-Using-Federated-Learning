import tensorflow as tf
from keras import layers #try tensorflow.keras if there are issues
from keras.models import Model
from keras.layers import Input, Conv2D
import segmentation_models as sm

class Unet(tf.keras.Model):
    def __init__(self, backbone, encoder_weights=None, name="custom_unet", **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        
        self.base_model = sm.Unet(backbone, encoder_weights=encoder_weights)
        
        self.input_layer = Input(shape=(None, None, 1))
        self.conv_layer = Conv2D(3, (1,1))
        
    def call(self, inputs):
        x = self.conv_layer(inputs)
        return self.base_model(x)

    def compile_model(self):
        self.compile(
            'adam',
            loss=sm.losses.dice_loss,
            metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        )

def train(model, training_generator, validation_generator, set_epochs, log_dir='./logs'):
    model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=set_epochs,
        verbose=2,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    )

def test(model, test_generator):
    """
    Evaluate the given model on the test data.

    Parameters:
    - model: The trained model to be evaluated.
    - test_generator: Data generator for testing.

    Returns:
    - Evaluation metrics (loss, IOU Score, and FScore).
    """
    evaluation_metrics = model.evaluate(test_generator, verbose=2)
    return evaluation_metrics

