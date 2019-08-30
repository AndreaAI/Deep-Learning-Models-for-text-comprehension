# Vectorize sentence+triples, questions and answers
def vectorize_sent_triple_all(data_sent, data_triple, questions, entities, word_idx, sentence_size, memory_size):
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
    
    for story in data_sent:
      story_corresp = []
      for word in story[0]:
        if word in ent_in_questions and word!= "film":
          story_corresp = story
          
      for i, sentence in enumerate(story_corresp, 1):
        ls = max(0, sentence_size - len(sentence))
        ss.append([word_idx[w] for w in sentence] + [0] * ls)
        dd.append([word_idx[w] for w in sentence] + [0] * ls)
        
    if story_corresp == []:
      for story in data_triple:
        for word in story[0][0]:
          if word in ent_in_questions and word!= "film":
            story_corresp = story
            
        for key, value in story_corresp:
          value = [value]
          ls = max(0, sentence_size - len(key))
          ld = max(0, sentence_size - len(value))
          ss.append([word_idx[k] for k in key] + [0] * ls)
          dd.append([word_idx[v] for v in value] + [0] * ld)
          
    # Save only the sentences that fit in memory
    ss = ss[::-1][:memory_size]
    dd = dd[::-1][:memory_size]
    
    # Fill in until memory size
    lm = max(0, memory_size - len(ss))
    ln = max(0, memory_size - len(dd))
    
    for _ in range(lm):
      ss.append([0] * sentence_size)
    for _ in range(ln):
      dd.append([0] * sentence_size)
      
    K.append(ss)
    V.append(dd)
    Q.append(q)
    A.append(y)
    
  return np.array(K), np.array(V), np.array(Q), np.array(A)
