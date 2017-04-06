import tensorflow as tf
import os

__so_dir = os.path.dirname(os.path.abspath(__file__))
__lib = tf.load_op_library(os.path.join(__so_dir, './__sampling/libsampling.so'))

single_negative_sampling = __lib.single_negative_sampling
multiple_negative_sampling = __lib.multiple_negative_sampling