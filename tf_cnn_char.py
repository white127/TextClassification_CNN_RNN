# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import random, time, os, datetime

#########################################################################
# 单层CNN文本分类模型
#########################################################################

#输入是定长序列，超过指定长度的截断，不足指定长度的补<a>
#构建字典
def build_vocab():
  code, vocab = int(0), {}
  vocab['UNKNOWN'] = code
  code += 1
  vocab['<a>'] = code
  code += 1
  for line in open('/export/jw/kg/data/sw_kgtrain.txt'):
    items = line.strip().split('\t')
    if len(items) != 3:
      continue
    for word in items[2].split('_'):
      if not word in vocab:
        vocab[word] = code
        code += 1
  return vocab

#分类名转id
def load_index():
  imap, c = {}, int(0)
  for line in open('/export/jw/kg/data/sw_kgtrain.txt'):
    items = line.strip().split('\t')
    if not imap.has_key(items[1]):
      imap[items[1]] = c
      c += 1
  return imap

#将分类结果转换成one-hot的形式
def encode_index(c, imap):
  index = imap[c]
  y = [int(0)] * len(imap)
  y[index] = int(1)
  return y

#be attention initialization of UNKNNOW
#对句子进行编码
def encode_sent(vocab, sent, size):
    x = []
    words = sent.split('_')
    for i in range(0, size):
        if i < len(words):
          if words[i] in vocab:
            x.append(vocab[words[i]])
          else:
            x.append(vocab['UNKNOWN'])
        else:
          x.append(vocab['<a>'])
    return x

#读取验证数据,验证数据格式和训练数据一样
def load_data_val(testList, vocab, index, batch_size, sent_len, imap):
    xlist, ylist, origxlist = [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        c, s = testList[true_index]
        xlist.append(encode_sent(vocab, s, sent_len))
        ylist.append(encode_index(c, imap))
        origxlist.append(s)
    return np.array(xlist, dtype='float32'), np.array(ylist, dtype='float32'), origxlist

def load_train_list():
  tmap, tlist = {}, []
  for line in open('/export/jw/kg/data/sw_kgtrain.txt'):
    items = line.strip().split('\t')
    if (len(items) == 2):
      items.append('')
    if not tmap.has_key(items[1]):
      tmap[items[1]] = []
    tmap[items[1]].append(items[2])
    tlist.append((items[1], items[2]))
  return tmap, tlist

def load_data(train_list, vocab, batch_size, sent_len, imap):
  xlist, ylist = [], []
  for i in xrange(0, batch_size):
    c, sent = train_list[random.randint(0, len(train_list) - 1)]
    xlist.append(encode_sent(vocab, sent, sent_len))
    ylist.append(encode_index(c, imap))
  return np.array(xlist, dtype='float32'), np.array(ylist, dtype='float32')

class CNN(object):
  def __init__(
    self, sequence_length, batch_size,
    vocab_size, embedding_size,
    filter_sizes, num_filters, num_classes, l2_reg_lambda=0.0):

    #用户问题,字向量使用embedding_lookup
    self.x_batch = tf.placeholder(tf.int32, [batch_size, sequence_length], name="x_batch")
    self.y_batch = tf.placeholder(tf.int32, [batch_size, num_classes], name='y_batch')
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    print("xlist", self.x_batch)

    # Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      We = tf.Variable(
        tf.truncated_normal([vocab_size, embedding_size], stddev=0.1),
        name="W")
    chars = tf.nn.embedding_lookup(We, self.x_batch)
    self.embedded_chars = chars
    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W-%s" % filter_size)
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b-%s" % filter_size)
        conv = tf.nn.conv2d(
          self.embedded_chars_expanded,
          W,
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="conv"
        )
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
          h,
          ksize=[1, sequence_length - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="pool"
        )
        pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 3), [-1, num_filters_total])
    #dropout
    h_drop = tf.nn.dropout(pooled_reshape, self.dropout_keep_prob)

    Wfc = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='Wfc')
    bfc = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bfc')
    h_output = tf.nn.xw_plus_b(h_drop, Wfc, bfc, name='scores')
    print('h_output', h_output)

    with tf.name_scope("output"):
      self.orig_y = tf.argmax(self.y_batch, 1)
      self.pred_y = tf.argmax(h_output, 1)

    with tf.name_scope("loss"):
      #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h_output, labels=self.y_batch)
      #print('batch_loss', cross_entropy)
      #self.loss = tf.reduce_mean(cross_entropy)
      #print('loss ', self.loss)
      self.loss = tf.losses.softmax_cross_entropy(self.y_batch, h_output)

      # Accuracy
    with tf.name_scope("accuracy"):
      correct = tf.equal(tf.argmax(h_output, 1), tf.argmax(self.y_batch, 1))
      print('correct', correct)
      self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
sent_len = int(100)

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================
# Load data
print("Loading data...")

def train_step(x_batch, y_batch):
    feed_dict = {
        cnn.x_batch: x_batch,
        cnn.y_batch: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

def dev_step(testList, vocab, batch_size, sent_len, imap):
    index, output_list, origy_list, origx_list = int(0), [], [], []
    while True:
        x_batch, y_batch, origx = load_data_val(testList, vocab, index, batch_size, sent_len, imap)
        feed_dict = {cnn.x_batch: x_batch, cnn.y_batch: y_batch, cnn.dropout_keep_prob: 1.0}
        origy, output = sess.run([cnn.orig_y, cnn.pred_y], feed_dict)
        for c in output:
            output_list.append(c)
        for c in origy:
            origy_list.append(c)
        for c in origx:
            origx_list.append(c)
        index += batch_size
        if index >= len(testList):
            break
    fp = file('/export/jw/kg/cnn.output', 'w+')
    i2nmap = {}
    for name, index in imap.items():
      i2nmap[index] = name
    for i in xrange(0, len(output_list)):
        fp.write(i2nmap[int(output_list[i])] + '\t' + i2nmap[origy_list[i]] + '\t' + origx_list[i] + '\n')
    fp.close()
    print 'wirte done ......'

def load_test_list():
  testList = []
  for line in open('/export/jw/kg/data/sw_kgval.txt'):
    items = line.strip().split('\t')
    if (len(items) == 2):
      items.append('')
    testList.append((items[1], items[2]))
  return testList

vocab = build_vocab()
train_map, train_list = load_train_list()
test_list = load_test_list()
imap = load_index()
xlist, ylist = load_data(train_list, vocab, FLAGS.batch_size, sent_len, imap)
num_classes = ylist.shape[1]
print("Load done...")

# Training
# ==================================================

with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CNN(
                    sequence_length=sent_len,
                    batch_size=FLAGS.batch_size,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    num_classes=num_classes,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.RMSPropOptimizer(0.0005)
            #optimizer = tf.train.AdamOptimizer(0.0001)
            #optimizer = tf.train.GradientDescentOptimizer(1e-2)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            #train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            # Training loop. For each batch...
            for i in range(FLAGS.num_epochs):
                try:
                    x_batch, y_batch = load_data(train_list, vocab, FLAGS.batch_size, sent_len, imap)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(test_list, vocab, FLAGS.batch_size, sent_len, imap)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                except Exception as e:
                    print(e)

