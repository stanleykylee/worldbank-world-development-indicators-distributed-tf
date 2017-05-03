import tensorflow as tf

# implementation based on https://github.com/ischlag/distributed-tensorflow-example

tf.app.flags.DEFINE_string("role", "", "ps/worker")
tf.app.flags.DEFINE_string("param_servers", "", "parameter servers")
tf.app.flags.DEFINE_string("worker_servers", "", "worker servers")
tf.app.flags.DEFINE_integer("task_index", 0, "index of task")

FLAGS = tf.app.flags.FLAGS

param_servers = FLAGS.param_servers.split(",")
worker_servers = FLAGS.worker_servers.split(",")

cluster = tf.train.ClusterSpec({ "ps": param_servers, "worker": worker_servers });

server = tf.train.Server(cluster, FLAGS.role, FLAGS.task_index)

if FLAGS.role == "ps":
    server.join()
elif FLAGS.role == "worker":
    with tf.device(tf.train.replica_device_setter(
	worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
        print "hello tf!"
