### Functions related to sentence format representation

# Parse information in sentence format
def parse_arguments(lines, ent_list, ent_rev):

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
            
      if len(line)<2:
        data.append(story)
            
      if len(line)>2:	
        sent = tokenize_movies(line, ent_list, ent_rev)
        story.append(sent)
        
    return data

# Vectorize info, questions and anwers
def vectorize_all(data, questions, entities, word_idx, sentence_size, memory_size):
  S = [] # Story
  Q = [] # Question
  A = [] # Answer  
  
  print("Vectorizing movies")
  for query, answer in questions:
    ss = []
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
      
      for word in story[0]:
        if word in ent_in_questions and word!= "film":
          story_corresp = story
          
      for i, sentence in enumerate(story_corresp, 1):
        ls = max(0, sentence_size - len(sentence))
          ss.append([word_idx[w] for w in sentence] + [0] * ls)
        
    # Save only the sentences that fit in the memory
    ss = ss[::-1][:memory_size]
    
    # Fill in until memory size
    lm = max(0, memory_size - len(ss))
    for _ in range(lm):
      ss.append([0] * sentence_size)
      
    S.append(ss)
    Q.append(q)
    A.append(y)
    
  return np.array(S), np.array(Q), np.array(A)

def tokenize_movies(sent, entities_list, ent_rev):
  for ent, idx in entities_list:
    sent = ent.sub(idx, sent)
    sent = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    
  for idx, s in enumerate(sent):
    for k, v in ent_rev.items():
      if s == k:
        s = v
        sent[idx] = v
        
  return sent
