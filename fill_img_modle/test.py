import tensorflow as tf
# manually put back imported modules
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")
with tf.Session() as sess:
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
  open("converteds_model.tflite", "wb").write(tflite_model)

# import tensorflow as tf
#
# img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
# var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
# val = img + var
# out = tf.identity(val, name="out")
#
# with tf.Session() as sess:
#   converter = tf.contrib.lite.toco_convert.from_session(sess, [img], [out])
#   tflite_model = converter.convert()
#   open("converted_model.tflite", "wb").write(tflite_model)
