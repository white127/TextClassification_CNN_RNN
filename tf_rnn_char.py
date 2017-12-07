import tensorflow as tf
import numpy as np
import random, datetime

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

def load_index():
  imap, c = {}, int(0)
  for line in open('/export/jw/kg/data/sw_kgtrain.txt'):
    items = line.strip().split('\t')
    if not imap.has_key(items[1]):
      imap[items[1]] = c
      c += 1
  return imap

def encode_index(c, imap):
  index = imap[c]
  y = [int(0)] * len(imap)
  y[index] = int(1)
  return y

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

def encode_mask(sent, size):
    mask = []
    words = sent.split('_')
    for i in range(0, size):
        if i < len(words):
            mask.append(1)
        else:
            mask.append(0)
    return mask

def load_data_val(testList, vocab, index, batch_size, sent_len, imap):
    xlist, ylist, mask_x, origxlist = [], [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        c, s = testList[true_index]
        xlist.append(encode_sent(vocab, s, sent_len))
        ylist.append(encode_index(c, imap))
        origxlist.append(s)
        mask_x.append(encode_mask(s, sent_len))
    return np.array(xlist, dtype='float32'), np.array(ylist, dtype='float32'), np.transpose(np.array(mask_x, dtype='float32')), origxlist

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

def load_test_list():
  testList = []
  for line in open('/export/jw/kg/data/sw_kgval.txt'):
    items = line.strip().split('\t')
    if (len(items) == 2):
      items.append('')
    testList.append((items[1], items[2]))
  return testList

def load_train_data(train_list, vocab, batch_size, sent_len, imap):
   xlist, ylist, mask_x = [], [], []
   for i in range(0, batch_size):
       c, sent = train_list[random.randint(0, len(train_list) - 1)]
       xlist.append(encode_sent(vocab, sent, sent_len))
       ylist.append(encode_index(c, imap))
       mask_x.append(encode_mask(sent, sent_len))
   return np.array(xlist, dtype='float32'), np.array(ylist, dtype='float32'), np.transpose(np.array(mask_x, dtype='float32'))

class RNN_Model(object):
    def __init__(self,config,is_training=True):
        self.keep_prob=config.keep_prob
        self.batch_size=config.batch_size
        num_step=config.num_step

        self.input_data = tf.placeholder(tf.int32, [self.batch_size, num_step])
        self.target = tf.placeholder(tf.int64, [self.batch_size, config.num_classes])
        self.mask_x = tf.placeholder(tf.float32, [num_step, self.batch_size])

        num_classes=config.num_classes
        hidden_neural_size=config.hidden_neural_size
        vocabulary_size=config.vocabulary_size
        embed_dim=config.embed_dim
        hidden_layer_num=config.hidden_layer_num

        #fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
        fw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_neural_size, activation=tf.nn.relu)
        if self.keep_prob<1:
            fw_cell =  tf.contrib.rnn.DropoutWrapper(
                fw_cell,output_keep_prob=self.keep_prob
            )
        self._initial_state = fw_cell.zero_state(self.batch_size,dtype=tf.float32)
        #bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
        bw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_neural_size, activation=tf.nn.relu)
        if self.keep_prob<1:
            bw_cell =  tf.contrib.rnn.DropoutWrapper(
                bw_cell,output_keep_prob=self.keep_prob
            )
        self._initial_state = bw_cell.zero_state(self.batch_size,dtype=tf.float32)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,self.input_data)

        if self.keep_prob<1:
            inputs = tf.nn.dropout(inputs,self.keep_prob)

        """
        out_put=[]
        state=self._initial_state
        print state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step>0: tf.get_variable_scope().reuse_variables()
                (cell_output,state)=cell(inputs[:,time_step,:],state)
                out_put.append(cell_output)
        out_put = out_put * self.mask_x[:,:,None]
        """
        state = self._initial_state
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, embed_dim])
        inputs = tf.split(inputs, num_step)
        out_put, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, initial_state_fw=state, initial_state_bw=state)
        out_put = out_put * self.mask_x[:, :, None]

        with tf.name_scope("mean_pooling_layer"):
            out_put = tf.reduce_sum(out_put,0) / (tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[hidden_neural_size * 2, num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[num_classes],dtype=tf.float32)
            self.logits = tf.matmul(out_put, softmax_w) + softmax_b

        with tf.name_scope("loss"):
            #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10, labels=self.target)
            self.loss = tf.losses.softmax_cross_entropy(self.target, self.logits)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

        with tf.name_scope("output"):
            self.orig_y = tf.argmax(self.target, 1)
            self.pred_y = tf.argmax(self.logits, 1)

tf.flags.DEFINE_integer('evaluate_every',1000,'evaluate every')
tf.flags.DEFINE_integer('batch_size',128,'the batch_size of the training procedure')
tf.flags.DEFINE_float('lr',0.1,'the learning rate')
tf.flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
tf.flags.DEFINE_integer('emdedding_dim',100,'embedding dim')
tf.flags.DEFINE_integer('hidden_neural_size',100,'LSTM hidden neural size')
tf.flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
tf.flags.DEFINE_integer('max_len',100,'max_len of training sentence')
tf.flags.DEFINE_float('init_scale',0.1,'init scale')
tf.flags.DEFINE_float('keep_prob',0.5,'dropout rate')
tf.flags.DEFINE_integer('num_epoch',100000,'num epoch')
tf.flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

vocab = build_vocab()
train_map, train_list = load_train_list()
test_list = load_test_list()
imap = load_index()
x, y, mask_x = load_train_data(train_list, vocab, FLAGS.batch_size, FLAGS.max_len, imap)

class Config(object):
    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=len(vocab)
    embed_dim=FLAGS.emdedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    keep_prob=FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    num_classes = len(imap)

config = Config()
eval_config=Config()
eval_config.keep_prob=1.0

with tf.Graph().as_default():
    with tf.device('/gpu:0'):
      session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,1*FLAGS.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = RNN_Model(config=config,is_training=True)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            dev_model = RNN_Model(config=eval_config,is_training=False)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(0.005)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def train_step(model, x, y, mask_x):
            fetches = [model.cost, model.accuracy, global_step, train_op]
            feed_dict = {
                model.input_data : x,
                model.target : y,
                model.mask_x : mask_x
            }
            #state = sess.run(model._initial_state)
            #print state
            #print model._initial_state
            #for i , (c,h) in enumerate(model._initial_state):
            #feed_dict[c]=state.c
            #feed_dict[h]=state.h
            cost, accuracy, step, _ = sess.run(fetches, feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))

        def dev_step(model, testList, vocab, batch_size, sent_len, imap):
            index, output_list, origy_list, origx_list = int(0), [], [], []
            while True:
                x, y, mask_x, origx = load_data_val(testList, vocab, index, batch_size, sent_len, imap)
                feed_dict = {model.input_data : x, model.target : y, model.mask_x: mask_x}
                origy, output = sess.run([model.orig_y, model.pred_y], feed_dict)
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

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        for i in range(config.num_epoch):
            x, y, mask_x = load_train_data(train_list, vocab, FLAGS.batch_size, FLAGS.max_len, imap)
            train_step(model, x, y, mask_x)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_step(dev_model, test_list, vocab, FLAGS.batch_size, FLAGS.max_len, imap)

