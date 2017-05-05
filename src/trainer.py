from __future__ import print_function
import argparse
import sys

import tensorflow as tf
import random

FLAGS = None


class PopulationData(object):
    def __init__(self, source_data, dataset):
        self.data = []
        self.labels = []
        self.seqlen = []
        src_data = []
        if dataset == "train":
            src_data = source_data[:int((len(source_data)+1)*.80)]
        elif dataset == "test":
            src_data = source_data[int(len(source_data)*.80+1):]
        i = 0
        for row in src_data:
            self.seqlen.append(24)
            label = row.pop(-1)
            self.data.append(row)
            if label[0] < 5:
                self.labels.append([1., 0.])
            else:
                self.labels.append([0., 1.])
            i += 1
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


def main(_):
    def dynamic_rnn(x, seqlen, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, seq_max_len, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            global_step = tf.contrib.framework.get_or_create_global_step()

            # ==========
            #   MODEL
            # ==========
            # Data sets
            file = open("population-data.csv", "r")
            data = list()
            for line in file:
                data.append([[float(x)] for x in line.split(',')])
            file.close()
            random.shuffle(data)

            # Parameters
            learning_rate = 0.01
            training_iters = 1000000
            batch_size = 128
            display_step = 10

            # Network Parameters
            seq_max_len = 24  # Sequence max length
            n_hidden = 64  # hidden layer num of features
            n_classes = 2  # linear sequence or not

            trainset = PopulationData(source_data=data, dataset="train")
            testset = PopulationData(source_data=data, dataset="test")

            # tf Graph input
            x = tf.placeholder("float", [None, seq_max_len, 1])
            y = tf.placeholder("float", [None, n_classes])
            # A placeholder for indicating each sequence length
            seqlen = tf.placeholder(tf.int32, [None])

            # Define weights
            weights = {
                'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

            pred = dynamic_rnn(x, seqlen, weights, biases)

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            init = tf.global_variables_initializer()

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(init)
                step = 1
                # Keep training until reach max iterations
                while step * batch_size < training_iters:
                    batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                    # Run optimization op (backprop)
                    mon_sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                                       seqlen: batch_seqlen})
                    if step % display_step == 0:
                        # Calculate batch accuracy
                        acc = mon_sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                                seqlen: batch_seqlen})
                        # Calculate batch loss
                        loss = mon_sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                                             seqlen: batch_seqlen})
                        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss) + ", Training Accuracy= " +
                              "{:.5f}".format(acc))
                    step += 1
                print("Optimization Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)