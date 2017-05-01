import tensorflow as tf

tf.app.flags.DEFINE_string("role", "", "ps/worker")
tf.app.flags.DEFINE_string("param_servers", "", "parameter servers")
tf.app.flags.DEFINE_string("worker_servers", "", "worker servers")
tf.app.flags.DEFINE_integer("task_index", 0, "index of task")

FLAGS = tf.app.flags.FLAGS

cluster = tf.train.ClusterSpec({ "ps": FLAGS.param_servers, "worker": FLAGS.worker_servers }); 

server = tf.train.Server(cluster, FLAGS.role, FLAGS.task_index) 
