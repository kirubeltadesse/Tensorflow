# create Tensorflow pipeline

# Create the data
import numpy as np
import tensorflow as tf

x_input = np.random.sample((1,2))

# using a placeholder
x = tf.placeholder(tf.float32, shape=[1,2], name = 'X')

# define the dataset method
dataset = tf.data.Dataset.from_tensor_slices(x)

# create the pipeline
## make initialize the pipeline
## create iterator

iterator = dataset.make_initializable_iterator()
# note in this case there are only two points
get_next = iterator.get_next()


# Execute the operation 
with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iterator.initializer, feed_dict={ x: x_input})
    print(sess.run(get_next))
