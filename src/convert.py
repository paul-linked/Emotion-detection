import tensorflow as tf

new_model= tf.keras.models.load_model(filepath="C:\\Users\\PLundberg\\Desktop\\damn\\model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()
open("tf_lite_model.tflite", "wb").write(tflite_model)
