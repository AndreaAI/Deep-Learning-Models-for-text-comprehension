### Execution example of the model KVMemNN over movies dataset.
from __future__ import absolute_import
from __future__ import print_function

from data_processing import load_movies_questions, load_movies_sentences, load_movies_windows, load_movies_triples
from data_processing_sentences import vectorize_all
from data_processing_triples import vectorize_triples_all
from data_processing_windows import vectorize_windows_all

from sklearn import cross_validation, metrics
from kvmemnn import MemNN_KV
from itertools import chain
from six.moves import range

import tensorflow as tf
import numpy as np
from kvmemnn import zero_nil_slot, add_gradient_noise

tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("lrate_decay_steps", 10000, "Steps decay learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 10, "Batch size for training.")
tf.flags.DEFINE_integer("feature_size", 20, "Feature size")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 40, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 80, "Maximum size of memory.")
tf.flags.DEFINE_string("level", "sentence", "Preprocess of the data (sentence, window, triple)")
tf.flags.DEFINE_string("data_dir", "data/peliculas_final/", "Directory containing Movies")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("output_file", "single_scores.csv", "Name of output file for final bAbI accuracy scores.")
tf.flags.DEFINE_integer("window_size", 8, "Window size.")

FLAGS = tf.flags.FLAGS

print("Starting execution about Movies:")
starting_point2 = time.time()

train_questions, val_questions, test_questions, entities = load_movies_questions(FLAGS.data_dir)
questions = train_questions +  val_questions + test_questions

vocab2 = sorted(reduce(lambda x, y: x | y, (set(list(a.rsplit(', ') + q)) for q, a in questions)))
vocab5 = sorted(reduce(lambda x, y: x | y, (set(list([a.lstrip()])) for q, a in questions)))
vocab = sorted(list(set(vocab2+vocab5+entities)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

vocab_size = len(word_idx) + 1 # +1 for nil word
query_size = max(map(len, (q for q, _ in questions)))


# train/validation/test sets

if FLAGS.level == "sentence":
  data = load_movies_sentences(FLAGS.data_dir)
  
  vocab1 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s))) for s in data)))
  vocab_sentence = sorted(list(set(vocab1+vocab)))

  word_idx = dict((c, i + 1) for i, c in enumerate(vocab_sentence))
  
  vocab_size = len(word_idx) + 1 # +1 for nil word
  max_story_size = max(map(len, (s for s in data)))
  print("max_story_size", max_story_size)
  mean_story_size = int(np.mean(map(len, (s for s in data))))
  sentence_size = max(map(len, chain.from_iterable(s for s in data)))
  memory_size = min(FLAGS.memory_size, max_story_size)
  sentence_size = max(query_size, sentence_size) # for the position

  print("Longest sentence length", sentence_size)
  print("Longest story length", max_story_size)
  print("Average story length", mean_story_size)

  S, Q, A = vectorize_all(data, train_questions, entities, word_idx, sentence_size, memory_size)

  if Q.shape[0]-S.shape[0] > 0:
    add = np.zeros((Q.shape[0]-S.shape[0],S.shape[1],S.shape[2]),  np.int32)
    S = np.concatenate((S, add), axis=0)

  trainS = S
  trainQ = Q
  trainA = A
    
  # When considering sentence format, key and value components store the same info
  trainK = trainS
  trainV = trainS

  S, Q, A = vectorize_all(data, val_questions, entities, word_idx, sentence_size, memory_size)

  if Q.shape[0]-S.shape[0] > 0:
    add = np.zeros((Q.shape[0]-S.shape[0],S.shape[1],S.shape[2]),  np.int32)
    S = np.concatenate((S, add), axis=0)

  valS = S
  valQ = Q
  valA = A


  valK = valS
  valV = valS


   S, Q, A = vectorize_all(data, test_questions, entities, word_idx, sentence_size, memory_size)

  if Q.shape[0]-S.shape[0] > 0:
    add = np.zeros((Q.shape[0]-S.shape[0],S.shape[1],S.shape[2]),  np.int32)
    S = np.concatenate((S, add), axis=0)

  testS = S
  testQ = Q
  testA = A

  testK = testS
  testV = testS

  print("Training set shape", trainS.shape)

  # params
  n_train = trainS.shape[0]
  n_test = testS.shape[0]
  n_val = valS.shape[0]

  print("Training Size", n_train)
  print("Validation Size", n_val)
  print("Testing Size", n_test)


if FLAGS.level == "window":

  window_size = FLAGS.window_size 
  windows = load_movies_windows(FLAGS.data_dir)

  vocab4 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(chain.from_iterable((a) for a in s)  ))) for s in windows)))
  vocab_window = sorted(list(set(vocab+vocab4)))

  word_idx = dict((c, i + 1) for i, c in enumerate(vocab_window))
  vocab_size = len(word_idx) + 1 # +1 for nil word
    
  max_story_window_size = max(map(len, (s for s in windows)))
  sentence_size = max(map(len, chain.from_iterable(s for s in windows)))
  print("max_story_window_size", max_story_window_size)
  sentence_size = max(query_size, sentence_size) # for the position
  memory_size = min(FLAGS.memory_size, max_story_window_size)

  print("Longest story length", max_story_window_size)
          
  max_window_size = window_size
  mean_window_size = int(np.mean(map(len, chain.from_iterable(chain.from_iterable(w for w in s[0]) for s in windows))))

  print("Longest window length", max_window_size)
  print("Average window length", mean_window_size)

  K, V, Q, A = vectorize_window_todo(windows, train_questions, entities, word_idx, window_size, sentence_size, memory_size)

  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)
 
  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)

  trainK = K
  trainV = V
  trainQ = Q
  trainA = A

  K, V, Q, A = vectorize_window_todo(windows, val_questions, entities, word_idx, window_size, sentence_size, memory_size)

  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)

 
  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)

  valK = K
  valV = V
  valQ = Q
  valA = A

  K, V, Q, A = vectorize_window_all(windows, test_questions, entities, word_idx, window_size, sentence_size, memory_size)


  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)

  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)
 
  testK = K
  testV = V
  testQ = Q
  testA = A

  print("Training set shape", trainK.shape)
  # params
  n_train = trainK.shape[0]
  n_test = testK.shape[0]
  n_val = valK.shape[0]
  print("Training Size", n_train)
  print("Validation Size", n_val)
  print("Testing Size", n_test)


if FLAGS.level == "triple":
  triples = load_movies_triples(FLAGS.data_dir)

  vocab3 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(chain.from_iterable(([a[0]]) for a in s)  ))) for s in triples)))
  vocab6 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(([a[1]]) for a in s)  )) for s in triples)))
  vocab_triple = sorted(list(set(vocab+vocab3+vocab6)))

  word_idx = dict((c, i + 1) for i, c in enumerate(vocab_triple))

  vocab_size = len(word_idx) + 1 # +1 for nil word

  max_story_triple_size = max(map(len, (s for s in triples)))
  sentence_size = max(map(len, chain.from_iterable(s for s in triples)))
  print("max_story_size", max_story_triple_size)
  sentence_size = max(query_size, sentence_size) # for the position
  memory_size = min(FLAGS.memory_size, max_story_triple_size)

  max_obj_size = max(map(len, chain.from_iterable(chain.from_iterable(w for _,w in s) for s in triples)))
  print("max_obj_size", max_obj_size)

  K, V, Q, A = vectorize_triple_all(triples, train_questions, entities, word_idx, sentence_size, memory_size)

  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)


  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)

  trainK = K
  trainV = V
  trainQ = Q
  trainA = A

  K, V, Q, A = vectorize_triple_all(triples, val_questions, entities, word_idx, sentence_size, memory_size)

  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)

  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)

  valK = K
  valV = V
  valQ = Q
  valA = A

  K, V, Q, A = vectorize_triple_all(triples, test_questions, entities, word_idx, sentence_size, memory_size)


  if Q.shape[0]-K.shape[0] > 0:
    add = np.zeros((Q.shape[0]-K.shape[0],K.shape[1],K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=0)
    add = np.zeros((Q.shape[0]-V.shape[0],V.shape[1],V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=0)


  if Q.shape[1]-K.shape[2] > 0:
    add = np.zeros((K.shape[0],K.shape[1],Q.shape[1]-K.shape[2]),  np.int32)
    K = np.concatenate((K, add), axis=2)
    add = np.zeros((V.shape[0],V.shape[1],Q.shape[1]-V.shape[2]),  np.int32)
    V = np.concatenate((V, add), axis=2)
 
  testK = K
  testV = V
  testQ = Q
  testA = A

 print("Training memory key shape", trainK.shape)

  # params
  n_train = trainK.shape[0]
  n_test = testK.shape[0]
  n_val = valK.shape[0]
  print("Training Size", n_train)
  print("Validation Size", n_val)
  print("Testing Size", n_test)


if FLAGS.level == "sentencetriple":

  data = load_movies_sentences(FLAGS.data_dir)
  triples = load_movies_triples(FLAGS.data_dir)
  
  vocab7 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s))) for s in data)))
  vocab_sentence = sorted(list(set(vocab7+vocab)))
  word_idx = dict((c, i + 1) for i, c in enumerate(vocab_sentence))
  vocab8 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(chain.from_iterable(([a[0]]) for a in s)))) for s in triples)))
  vocab9 = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(([a[1]]) for a in s) )) for s in triples)))
  vocab_triple = sorted(list(set(vocab+vocab7+vocab8+vocab9)))
  word_idx = dict((c, i + 1) for i, c in enumerate(vocab_triple))
  vocab_size = len(word_idx) + 1 # +1 for nil word
  
  max_story_size = max(map(len, (s for s in data)))
  mean_story_size = int(np.mean(map(len, (s for s in data))))
  sentence_size = max(map(len, chain.from_iterable(s for s in data)))
  sentence_size = max(query_size, sentence_size) # for the position
  max_story_triple_size = max(map(len, (s for s in triples)))
  memory_size = min(FLAGS.memory_size, max_story_size)

  K, V, Q, A = vectorize_sent_triple_all(data, triples, train_questions, entities, word_idx, sentence_size, memory_size)
  trainK = K
  trainV = V
  trainQ = Q
  trainA = A
  
  K, V, Q, A = vectorize_sent_triple_all(data, triples, val_questions, entities, word_idx, sentence_size, memory_size)
  valK = K
  valV = V
  valQ = Q
  valA = A
  
  K, V, Q, A = vectorize_sent_triple_todo(data, triples, test_questions, entities, word_idx, sentence_size, memory_size)
  testK = K
  testV = V
  testQ = Q
  testA = A
  
  # params
  n_train = trainK.shape[0]
  n_test = testK.shape[0]
  n_val = valK.shape[0]
  print("Training Size", n_train)
  print("Validation Size", n_val)
  print("Testing Size", n_test)
   
elapsed_time2 = time.time() - starting_point2

# Save results in files to visualize depending on the parameters values and choose the optimal set of parameters
with open("kvmemnn_time_movies_" + "_level" + str(FLAGS.level) + "_hops" + str(FLAGS.hops) + "_epochs" + str(FLAGS.epochs) + "_lrate" + str(FLAGS.learning_rate) + "_arate" + str(FLAGS.lrate_decay_steps) + "_batch" + str(FLAGS.batch_size) + "_eps" + str(FLAGS.epsilon) + "_feat" + str(FLAGS.feature_size) + "_emb" + str(FLAGS.embedding_size) + "_mem" + str(memory_size) + "_gradnorm" + str(FLAGS.max_grad_norm) + ".csv", 'a') as f:
  f.write('{}, {}, {}, {}\n'.format("Processing (segs)", "Processing (mins)", "Training (segs)", "Training (mins)"))


train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

batch_size = FLAGS.batch_size
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))


with tf.Graph().as_default():
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)

  global_step = tf.Variable(0, name="global_step", trainable=False)
  # decay learning rate
  starter_learning_rate = FLAGS.learning_rate
  decay_steps = FLAGS.lrate_decay_steps

  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20000, 0.96, staircase=True)
  
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  with tf.Session() as sess:
    model = MemNN_KV(batch_size=batch_size, vocab_size=vocab_size,
                          query_size=sentence_size, story_size=sentence_size, memory_key_size=memory_size,
                          feature_size=FLAGS.feature_size, memory_value_size=memory_size,
                          embedding_size=FLAGS.embedding_size, hops=FLAGS.hops, reader=FLAGS.reader,
                          l2_lambda=FLAGS.l2_lambda)
    grads_and_vars = optimizer.compute_gradients(model.loss_op)
    grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v) for g, v in grads_and_vars if g is not None]
    grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
    nil_grads_and_vars = []
    for g, v in grads_and_vars:
      if v.name in model._nil_vars:
        nil_grads_and_vars.append((zero_nil_slot(g), v))
      else:
        nil_grads_and_vars.append((g, v))

    train_op = optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=global_step)
    sess.run(tf.initialize_all_variables())

    def train_step(k, v, q, a):
      feed_dict = {
          model._memory_value: v,
          model._query: q,
          model._memory_key: k,
          model._labels: a,
          model.keep_prob: FLAGS.keep_prob
      }
      _, step, predict_op = sess.run([train_op, global_step, model.predict_op], feed_dict)
      return predict_op

    def test_step(k, v, q):
      feed_dict = {
          model._query: q,
          model._memory_key: k,
          model._memory_value: v,
          model.keep_prob: 1
      }
      preds = sess.run(model.predict_op, feed_dict)
      return preds

    for t in range(1, FLAGS.epochs+1):
      np.random.shuffle(batches)
      train_preds = []
      for start in range(0, n_train, batch_size):
        end = start + batch_size
        k = trainK[start:end]
        v = trainV[start:end]
        q = trainQ[start:end]
        a = trainA[start:end]

        predict_op = train_step(k, v, q, a)
        train_preds += list(predict_op)

  train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
  print('-----------------------')
  print('Epoch', t)
  print('Training Accuracy: {0:.2f}'.format(train_acc))
  print('-----------------------')
 

               
  if t % FLAGS.evaluation_interval == 0:
    # test on train dataset
    train_preds = test_step(trainK, trainV, trainQ)
    train_acc = metrics.accuracy_score(train_labels, train_preds)
    train_acc = '{0:.2f}'.format(train_acc)
    
    # eval dataset
    val_preds = test_step(valK, valV, valQ)
    val_acc = metrics.accuracy_score(val_labels, val_preds)
    val_acc = '{0:.2f}'.format(val_acc)
    
    # testing dataset
    test_preds = test_step(testK, testV, testQ)
    test_acc = metrics.accuracy_score(test_labels, test_preds)
    test_acc = '{0:.2f}'.format(test_acc)
    
    with open("kvmemnn_movies" + "_level" + str(FLAGS.level) +"_hops" + str(FLAGS.hops) + "_epochs" + str(FLAGS.epochs) +"_lrate" + str(FLAGS.learning_rate) + "_arate" +str(FLAGS.lrate_decay_steps) + "_batch" + str(FLAGS.batch_size) + "_eps" + str(FLAGS.epsilon) + "_feat" + str(FLAGS.feature_size) + "_emb" + str(FLAGS.embedding_size) + "_mem" + str(memory_size) + "_gradnorm" + str(FLAGS.max_grad_norm) + ".csv", 'a') as f:
      f.write('{}, {}, {}, {}\n'.format(t, train_acc, val_acc, test_acc))

  print('-----------------------')
  print('Epoch', t)
  print('Validation Accuracy:', val_acc)
  print('-----------------------')
  elapsed_time1 = time.time() - starting_point1
  
  
  with open("kvmemnn_tiempo_pelis" + "_level" + str(FLAGS.level) + "_hops" + str(FLAGS.hops) + "_epochs" + str(FLAGS.epochs) + "_lrate" +
  str(FLAGS.learning_rate) + "_arate" + str(FLAGS.lrate_decay_steps) + "_batch" + str(FLAGS.batch_size) + "_eps" + str(FLAGS.epsilon) + "_feat" + str(FLAGS.feature_size) + "_emb" + str(FLAGS.embedding_size) + "_mem" + str(memory_size) + "_gradnorm" + str(FLAGS.max_grad_norm) + ".csv", 'a') as f:
    f.write('{}, {}, {}, {}\n'.format(elapsed_time2, elapsed_time2/60, elapsed_time1, elapsed_time1/60))
