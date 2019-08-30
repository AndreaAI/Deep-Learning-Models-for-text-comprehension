### Functions to process entities and questions are independient of the text format that is being considered.
### There are specific functions to process the text depending on the format (raw text. windows, triples).

# Process entities file
def parse_entities(entities_files):
  ent_list = []
  re_list = []
  entities = {}
  ent_rev = {}
  print('Processing entities file...')
  with open(entities_files) as read:
    for l in read:
      l = l.rstrip()
      if len(l) > 0:
        ent_list.append(l)
  ent_list.sort(key=lambda x: -len(x))
  
  for i in range(len(ent_list)):
    k = ent_list[i]
    v = '__{}__'.format(i)
    entities[k] = v
    ent_rev[v] = k
    re_list = [(re.compile('\\b{}\\b'.format(re.escape(e))), '{}'.format(entities[e])) for e in ent_list]
    
return ent_rev, re_list


# Process questions file
def parse_questions(lines, ent_list, ent_rev):
    '''Parse questions provided in the Movies format'''

    print("Analyzing questions and answers")
    data=[]
    story = []
  for line in lines:
    if '\xef\xbb\xbf1' in line:
      line = line.replace("\xef\xbb\xbf1", "1")
      nid, line = line.split(' ', 1)
      nid = int(nid)
      if nid == 1:
        q, a = line.split('\t')
        q = tokenize_movies(q, ent_list, ent_rev)
        a = a.replace('\r', '')
        
        for ent, idx in ent_list:
          a = ent.sub(idx, a)
          a = a.replace('\n', '')
          
          for an in a:
            for k, v in ent_rev.items():
              if k in a:
                a = a.replace(k, v)
                break

        # remove question marks
        if q[-1] == "?":
          q = q[:-1]

        story.append('')
        data.append((q, a))

  return data
