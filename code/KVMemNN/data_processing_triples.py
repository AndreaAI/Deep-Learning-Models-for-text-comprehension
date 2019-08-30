### Functions related to triple format representation

# Parse information in triple format
def parse_triples(lines, ent_list, ent_rev):

  print("Analyzing triples")
  data = []
  story = []
  for line in lines:
    if len(line)>2:
      nid, line = line.split(' ', 1)
      nid = int(nid)
      if nid == 1:
        story = []
        
    if len(line)<2: # change of movie
      data.append(story)
      
    if len(line)>2:
      sent, obj = tokenize_triple(line, ent_list, ent_rev)
      if obj[-1] == ".":
        obj = obj[:-1]
        
    story.append((sent,obj))
    
  return data
  
  
# Vectorize triples, questions and answers
def vectorize_triple_todo(data, questions, entities, word_idx, sentence_size, memory_size):
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
        
    # Save only the triples that fit in the memory
    ss = ss[::-1][:memory_size]
    dd = dd[::-1][:memory_size]
    
    # Fill in until memory size
    lm = max(0, memory_size - len(ss))
    ln = max(0, memory_size - len(dd))
    
    for _ in range(lm):
      ss.append([0]*2)      
    for _ in range(ln):
      dd.append([0] * 1)
      
    K.append(ss)
    V.append(dd)
    Q.append(q)
    A.append(y)
    
return np.array(K), np.array(V), np.array(Q), np.array(A)


def tokenize_triple(sent, entities_list, ent_rev):
  for ent, idx in entities_list:
    sent = ent.sub(idx, sent)
    
  sent, b , c = sent.split(' ', 2)
  sent = sent + ' ' + b
  a = c.replace('\n', '')
  a = a.replace('\r', '')
  
  sent = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

  for k, v in ent_rev.items():
    if sent[0] == k:
      sent[0] = v
      break
  for k, v in ent_rev.items():
    if k in a:
      a = a.replace(k, v)
      
  return sent, a
