
import tensorflow as tf
GLOBAL_OFFSET = 50
THRESHOLD_GOLD = 0.8
RESIZE_DIM_256 = (256, 256)
RESIZE_DIM_224 = (224, 224)
NUM_SITES = 21 



def process_tf(image, c, d):
    """
    Process and augment a medical image for training.

    Args:
    image (Tensor): Input image.
    c (Tensor): Additional data associated with the image.
    d (Tensor): Another piece of data associated with the image.

    Returns:
    Tuple[Tensor, Tensor, Tensor]: Processed image and associated data.
    """
    image = tf.cast(image, tf.float32) / 255.0
    c = tf.cast(c, tf.int32)

    # Apply minimum threshold
    image = tf.minimum(image, THRESHOLD_GOLD)

    # Augmentation parameters
    flip = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0)
    delta_x = tf.random.uniform(shape=[], maxval=16, dtype=tf.int32, seed=8)
    delta_y = tf.random.uniform(shape=[], maxval=16, dtype=tf.int32, seed=5)

    ys = []
    for i in range(24):

        global_offset_dither = GLOBAL_OFFSET
        image_2d = image[global_offset_dither+delta_x:delta_x+240-global_offset_dither,
                         global_offset_dither+delta_y:delta_y+240-global_offset_dither,i ,:]
        if (flip == 1):
            image_2d = tf.image.flip_left_right(image_2d)

        image_2d = tf.image.resize(image_2d, RESIZE_DIM_256, preserve_aspect_ratio=False, antialias=False)
        ys.append(image_2d)

    ys = tf.convert_to_tensor(ys, dtype=tf.float32)
    return ys, c, d

def rotate_tf(image, c):
    """
    Rotate and augment a medical image for training.

    Args:
    image (Tensor): Input image.
    c (Tensor): Additional data associated with the image.

    Returns:
    Tuple[Tensor, Tensor]: Rotated and augmented image and associated data.
    """
    image = tf.cast(image, tf.float32) / 255.0
    c = tf.cast(c, tf.int32)
    image = tf.minimum(image, THRESHOLD_GOLD)

    # Augmentation parameters
    flip = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0)

    ys = []
    for i in range(8):
        image_2d = image[GLOBAL_OFFSET:-GLOBAL_OFFSET, GLOBAL_OFFSET:-GLOBAL_OFFSET, i, :]
        if flip == 1:
            image_2d = tf.image.flip_left_right(image_2d)
        image_2d = tf.image.resize(image_2d, RESIZE_DIM_224, preserve_aspect_ratio=False, antialias=False)
        ys.append(image_2d)

    ys = tf.convert_to_tensor(ys, dtype=tf.float32)
    return ys, c

def test_map(ys, z):
    """
    Prepare a medical image for testing.

    Args:
    ys (Tensor): Input image.
    z (Tensor): Additional data associated with the image.

    Returns:
    Tuple[Tensor, Tensor]: Processed image and associated data.
    """
    ys = tf.cast(ys, tf.float32) / 255.0
    z = tf.cast(z, tf.float32)
    ys = tf.minimum(ys, THRESHOLD_GOLD)
    ys = ys[GLOBAL_OFFSET:-GLOBAL_OFFSET, GLOBAL_OFFSET:-GLOBAL_OFFSET, :, :]

    yss = []
    for i in range(24):
        image_2d = tf.image.resize(ys[:, :, i, :], RESIZE_DIM_256, preserve_aspect_ratio=False, antialias=False)
        yss.append(image_2d)

    ys = tf.convert_to_tensor(yss, dtype=tf.float32)
    return ys, z
