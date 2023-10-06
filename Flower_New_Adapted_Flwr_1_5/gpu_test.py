import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
hasGPUSupport = tf.test.is_built_with_cuda()

gpuList = tf.config.list_physical_devices('GPU')

print("Tensorflow Compiled with CUDA/GPU Support:", hasGPUSupport)

print("Tensorflow can access", len(gpuList), "GPU")

print("Accessible GPUs are:")

print(gpuList) 
