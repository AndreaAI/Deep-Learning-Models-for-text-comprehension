### Functions related to window format representation

# Parse information in window format
def parse_windows(lines, ent_list, ent_rev):

  print("Analyzing movies")
  data = []
  story = []
  for line in lines:
    if '\xef\xbb\xbf1' in line:
      line = line.replace("\xef\xbb\xbf1", "1")
    if len(line)>2:
      nid, line = line.split(' ', 1)
    nid = int(nid)
    if nid == 1:
      story = []
      substory = None
      
    if len(line)<2:
      substory = [x for x in story if x]
      data.append(substory)
      
    if len(line)>2 and nid!=1:
      sent = tokenize_movies(line, ent_list, ent_rev)
      story.append(sent)
      
  max_length = None
  flatten = lambda data: reduce(lambda x, y: x + y, data)
  data = [(flatten(story)) for story in data if not max_length or len(flatten(story)) < max_length]
  
  story2 = []
  data2 = []
  for story in data:
    story2 = []
    for v in story:
      for k, vv in ent_rev.items():
        if v == vv:
          idx = story.index(v)
          if idx in range(3, len(story)-3):
            window = [story[idx-3], story[idx-2], story[idx-1], story[idx], story[idx+1], story[idx+2], story[idx+3]]
            center = v
            story2.append((window, center))
            
    data2.append(story2)

  return data2
  
# Vectorize windows, questions and answers  
def vectorize_window_todo(data, questions, entities, word_idx, sentence_size, window_size, memory_size):
  K = [] # Key
  V = [] # Value
  Q = [] # Question
  A = [] # Answer
  
  print("Vectorizing movies")
  for query, answer in questions:
    ss = []
    dd = []
    lq = max(0, sentence_size - len(query))
    q = [word_idx[w] for w in query] + [0] * lq
    y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
    y[word_idx[answer]] = 1
    
    ent_in_questions = []
    for ent in entities:
      if ent in query or ent in answer:
        ent_in_questions.append(ent)
        
    ent_in_questions = sorted(reduce(lambda x, y: x | y, (set(list([ent])) for ent in ent_in_questions)))
      
    for story in data:
      story_corresp = []
      for word in story[0][0]:
        if word in ent_in_questions and word!= "film":
          story_corresp = story
          
      for key, value in story_corresp:
        value = [value]
        ss.append([word_idx[k] for k in key])
        dd.append([word_idx[v] for v in value])
        
    # Save only the windows that fit in memory
    ss = ss[::-1][:memory_size]
    dd = dd[::-1][:memory_size]
    
    # Fill in until memory size
    lm = max(0, memory_size - len(ss))
    ln = max(0, memory_size - len(dd))
    
    for _ in range(lm):
      ss.append([0] * window_size)
    for _ in range(ln):
      dd.append([0] * 1)
      
    K.append(ss)
    V.append(dd)
    Q.append(q)
    A.append(y)
  
  return np.array(K), np.array(V), np.array(Q), np.array(A)
  
