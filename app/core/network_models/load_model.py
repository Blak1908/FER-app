
import os 
import tensorflow as tf 

def _load(checkpoint_path):
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(checkpoint_path).assert_consumed()
    return checkpoint


