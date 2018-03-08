# import alexnet
# import resnet_v1
import platform

import tensorflow as tf
import TFreader as TFRS_Reader
import vgg

slim=tf.contrib.slim
netname="VGG16"
rate=1
net=vgg.vgg_16
TFRECORDFILE="./dataset_1_to_6"
IS_PRETRAIN=True
PRETRAIN_FILEPATH="./vgg_16.ckpt"
def mymse(x,y):
    return tf.pow(x-y, 2)

def main(_dataset,model_path,log_dir):
    image,label=TFRS_Reader.PupilDataset(_dataset)
    # image.set_shape([120, 120, 3])
    images, labels = tf.train.batch([image, label], batch_size=32,capacity=5000)
    images = tf.image.resize_bicubic(images, [224, 224])

    predictions, end_points = net(images, num_classes=2)
    # predictions=tf.div(predictions,100)
    # Specify the loss function:
    tf.losses.mean_squared_error(predictions,labels)
    # slim.losses.mean_squared_error(predictions, labels)

    total_loss = tf.losses.get_total_loss()
    slim.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.00001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Specify where the Model, trained on ImageNet, was saved.
    # model_path =

    # Specify where the new model will live:
    # log_dir =

    # Restore only the convolutional layers:
    variables_to_restore = slim.get_variables_to_restore(exclude=[ 'vgg_16/fc6','vgg_16/fc7','vgg_16/fc8'])
    # variables_to_restore = slim.get_variables_to_restore(include=[ 'vgg_16/conv1','vgg_16/pool1','vgg_16/conv2','vgg_16/pool2'])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

    # values, indices=tf.nn.top_k(predictions, 1)
    # p=tf.one_hot(indices, 3, 1, 0)
    # p=tf.reshape(p,(32,3))
    # acc=slim.metrics.mean_squared_error(predictions, labels)
    # op1=slim.summary.scalar('summaries/acc/acc', acc)
    # op2=slim.summary.scalar('summaries/var/total_loss', total_loss)
    # op3=slim.summary.scalar('summaries/var/predictions', indices[0][0])
    # op4=slim.summary.scalar('summaries/var/labels', tf.nn.top_k(labels, 1)[1][0][0])
    op=slim.summary.merge_all(key='summaries')
    op=tf.Print(op,[total_loss,predictions,labels])
    # Start training.
    slim.learning.train(train_op, log_dir, init_fn=init_fn,save_summaries_secs=20,summary_op=op)
if __name__ == '__main__':
    if platform.system() == 'Windows':
        _dataset, _model, _log=["./"+TFRECORDFILE,PRETRAIN_FILEPATH,'./log']
    else:
        _dataset, _model, _log = [TFRECORDFILE, PRETRAIN_FILEPATH, "./log"]
    main(_dataset, _model, _log)