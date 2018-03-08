import tensorflow as tf
def PupilDataset(tf_record_filename="C:\\LPW\\train.tfrecords.AlexJ"):
    filename_queue = tf.train.string_input_producer([tf_record_filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'x': tf.FixedLenFeature([], tf.float32),
                                           'y': tf.FixedLenFeature([], tf.float32),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    lab = [tf.cast(features['x'], tf.float32),tf.cast(features['y'], tf.float32)]
    return img,lab