import sys,bisect

""" Global functions
"""

diff = 0.001

def floatGreater(f1, f2):
  f_diff = f1 - f2
  if floatEqual(f1, f2):
    return False
  else:
    if f_diff < 0:
      return False
  return True

def floatEqual(f1, f2):
  f_diff = f1 - f2
  if (abs(f_diff) < diff):
    return True
  return False

def floatCmp(f1, f2):
  f_diff = f1 - f2
  if (abs(f_diff)<diff):
    return 0
  elif (f_diff<0):
    return -1
  else:
    return 1

def square_distance(pointA, pointB):
    # squared euclidean distance
    distance = 0
    dimensions = len(pointA) # assumes both points have the same dimensions
    for dimension in range(dimensions):
        distance += (pointA[dimension] - pointB[dimension])**2
    return distance


""" Modified bisect_left of python for searching descending order     
"""
def bisect_left(a, x, lo=0, hi=None, 
      isGreater=(lambda x,y: x>y), reverse=False):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if reverse:
          if isGreater(a[mid], x): lo = mid+1
          else: hi = mid
        else:
          if isGreater(x,a[mid]): lo = mid+1
          else: hi = mid
    return lo

def bisect_right(a, x, lo=0, hi=None, 
      isGreater=(lambda x,y: x>y), reverse=False):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if reverse:
          if isGreater(x, a[mid]): hi = mid
          else: lo = mid+1
        else:
          if isGreater(a[mid], x): hi = mid
          else: lo = mid+1
    return lo

""" Nearest Neighbours List data structure for KDTree
    This is used for questions
"""

class MyQuestionList():
  def __init__(self, size):
    self.size = size
    self.q_list = []  # [(sd,q_id)]
    self.sd_val = {}  # {q_id:sd}
    self.sds = []     # store a redundant list of sds in a separate list
                      # for performance improvement
    self.radius = 0   # The searching radius should be infinite
                      # when the length of the nn list is still low 
  def pop(self, index):
    self.sds.pop(index)
    item = self.q_list.pop(index)
    del self.sd_val[item[1]]

  def insert(self, index, sd, q_id):
    if(len(self.q_list) == self.size):
      if (index == self.size):
        return
      else:
        self.pop(self.size-1)
       
    self.q_list.insert(index, (sd,q_id))
    self.sds.insert(index, sd)
    self.sd_val[q_id] = sd

  def get_sd_max(self):
    if (len(self.q_list) == 0):
      return 0
    if (not self.capacityReached()):
      return self.radius
    return self.sds[-1]

  def higherPriority(self, a, b):
    if floatGreater(a[0],b[0]):
      return False;
    elif floatEqual(a[0], b[0]):
      return (a[1] > b[1])
    else:
      return True

  def capacityReached(self):
    return len(self.q_list) == self.size
       
  def add(self, q_list0, sd0):
    if floatGreater(sd0, self.get_sd_max()) and self.capacityReached():
      return

    # searching radius is infinite when the size of 
    # nearest neighbours list is still low
    if not self.capacityReached():
      self.radius = max(self.radius, sd0)

    # Check if there're any existing q_id in the new list
    for q_id in list(q_list0):
      # check if the q_id was in the current list
      # to update the sd if needed
      if self.sd_val.get(q_id):
        sd = self.sd_val[q_id]
        if (floatGreater(sd,sd0)):
          # same question as that of the new one but having the greater sd, 
          # remove it out of the current list
          index = bisect_left(self.q_list, (sd, q_id), 
                    isGreater=self.higherPriority, reverse=True)
          self.pop(index)
        else:
          q_list0.remove(q_id)
    # No duplicated q_id is existing in the current best
    # Find a right place to put them in
    # Different from the topic list, this one should use bulk adding
    i = bisect_left(self.sds, sd0, isGreater=floatGreater)
    if (i == len(self.sds)) or (not floatEqual(self.sds[i], sd0)):
      q_list0.sort()
      for q_id in q_list0:
        self.insert(i, sd0, q_id)
    else:
      j = i + bisect_right(self.sds[i:], sd0, isGreater=floatGreater)
      # Merge new q_id list with sublist of q_ids from the index i and j
      k=0
      q_list0.sort(reverse=True)
      while (i<j) and (k<len(q_list0)):
        if (q_list0[k]>self.q_list[i][1]):
          self.insert(i, sd0, q_list0[k])
          k+=1
          i+=1
          j+=1
        else:
          i+=1
          
      if i==j:
       # there would be some item in q_list left
        for q_id in q_list0[k:]:
          self.insert(i, sd0, q_id)
          i+=1
      

  def get_best(self):
    return [item[1] for item in self.q_list]

""" Nearest Neighbours List data structure for KDTree
    This is used for topics
"""
class MyTopicList():
  def __init__(self, size):
    self.size = size
    self.t_list = [] # [(sd,t_id)]

  def get_sd_max(self):
    if (len(self.t_list) != 0):
      return self.t_list[-1][0]
    else:
      return 0

  def higherPriority(self, a, b):
    if floatGreater(a[0], b[0]):
      return False;
    elif floatEqual(a[0], b[0]):
      return (a[1] > b[1])
    else:
      return True 

  def insert(self, index, sd, t_id):
    if(len(self.t_list) == self.size):
      if (index == self.size):
        return
      else:
        self.t_list.pop(self.size-1) 
    self.t_list.insert(index, (sd, t_id))

  def add(self, sd, t_id):
    if floatGreater(sd,self.get_sd_max()) and (len(self.t_list)==self.size):
      return
    index = bisect_left(self.t_list, (sd, t_id), 
        isGreater=self.higherPriority, reverse=True)
    self.insert(index, sd, t_id)

  def get_best(self):
    return [item[1] for item in self.t_list]

""" 
Tue: Re-use KDTree k nearest neighbours (knn) implementation
     from the author Matej Drame, and fixed some bugs there
"""

""" Node class for KDTree
"""
class KDTreeNode():
    def __init__(self, value, left, right):
        self.value = value
        self.point = value.point
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return (self.left == None and self.right == None)

class KDTreeNeighbours():
    def __init__(self, query_point, t):
      self.query_point = query_point
      self.t_list = MyTopicList(t)

    def get_largest_distance(self):
      return self.t_list.get_sd_max()

    def add(self, node):
      sd = square_distance(node.point, self.query_point)
      self.t_list.add(sd, node.value.t_id)     
       
    def get_best(self):
      return self.t_list.get_best()

class KDTreeQuestionedNNs():
  def __init__(self, query_point, t):
      self.query_point = query_point
      self.t = t # total nearest questions wanted
      self.q_list = MyQuestionList(t)

  def get_largest_distance(self):
    return self.q_list.get_sd_max()

  def add(self, node):
    # no questions? no go
    sd = square_distance(node.point, self.query_point)
    if len(node.value.questions)==0:
      return  
    self.q_list.add(list(node.value.questions), sd)
  def get_best(self):
    return self.q_list.get_best()

class KDTree():
    """ - data is a list of topics
        - nodeListHandler is an internal data structure 
        to store topics or questions
    """
    def __init__(self, data):
        def build_kdtree(topic_list, depth):
            if not topic_list:
                return None

            # select axis based on depth so that axis cycles through all valid values
            axis = depth % len(topic_list[0].point) # assumes all points have the same dimension

            # sort point list and choose median as pivot point,
            # TODO: better selection method, linear-time selection, distribution
            topic_list.sort(cmp=floatCmp, key=lambda topic: topic.point[axis])
            median = len(topic_list)/2 # choose median

            # create node and recursively construct subtrees
            node = KDTreeNode(value=topic_list[median],
                              left=build_kdtree(topic_list[0:median], depth+1),
                              right=build_kdtree(topic_list[median+1:], depth+1))
            return node
        
        self.root_node = build_kdtree(data, depth=0)
         
    @staticmethod
    def construct_from_data(data):
        tree = KDTree(data)
        return tree

    def query(self, query_point, t=1, nodeListHandler=KDTreeNeighbours):
        def nn_search(node, query_point, depth, best_neighbours):
            if node == None:
                return
            # if we have reached a leaf, let's add to current best neighbours,
            # (if it's better than the worst one or if there is not enough neighbours)
            if node.is_leaf():
                best_neighbours.add(node)
                return
            
            # this node is no leaf
            
            # select dimension for comparison (based on current depth)
            axis = depth % len(query_point)

            # figure out which subtree to search
            near_subtree = None # near subtree
            far_subtree = None # far subtree (perhaps we'll have to traverse it as well)
            
            # compare query_point and point of current node in selected dimension
            # and figure out which subtree is farther than the other
            if floatGreater(node.point[axis], query_point[axis]):
                near_subtree = node.left
                far_subtree = node.right
            else:
                near_subtree = node.right
                far_subtree = node.left

            # recursively search through the tree until a leaf is found
            nn_search(near_subtree, query_point, depth+1, best_neighbours)

            # while unwinding the recursion, check if the current node
            # is closer to query point than the current best,
            # also, until t points have been found, search radius is infinity
            best_neighbours.add(node)
            
            # check whether there could be any points on the other side of the
            # splitting plane that are closer to the query point than the current best
            sd0 = (node.point[axis] - query_point[axis])**2 
            if not floatGreater(sd0, best_neighbours.get_largest_distance()):
                nn_search(far_subtree, query_point, depth+1, best_neighbours)
            
            return
        
        # if there's no tree, there's no neighbors
        if self.root_node != None:
            neighbours = nodeListHandler(query_point, t)
            nn_search(self.root_node, query_point, depth=0, best_neighbours=neighbours)
            result = neighbours.get_best()
        else:
            result = []
        
        return result

class topic:
  def __init__(self, t_id, t_x, t_y):
    self.t_id = t_id
    self.point = [t_x, t_y]
    self.questions = [] 

def getTopicFromLine(line):
    t_id, t_x, t_y = line.split(' ', 2)
    return topic(int(t_id), float(t_x), float(t_y))

def getQuestionFromLine(line):    
    arr = line.split(' ')
    return [arr[0], arr[1:]]

def printArray(arr):
    if len(arr) != 0:
      for i, e in enumerate(arr):
        if i == len(arr)-1:
          print e
        else:
          print e,

def readraw():
  line = raw_input()
  counts = line.split(' ', 2)
  t_count = int(counts[0])
  q_count = int(counts[1])
  query_count = int(counts[2])
  
  # reading the topics
  topic = None
  topic_list = []
  for i in xrange(t_count):
    line = raw_input()
    topic = getTopicFromLine(line)
    topic_list.append(topic)
  topicTree = KDTree.construct_from_data(topic_list)
  
  # sort topics first for looking up later
  topic_list.sort(key=lambda t: t.t_id)
  keys = [t.t_id for t in topic_list]
  
  # reading the questions
  q_id = None
  qtopic_dict = {} # a dictionary questioned topics
  arr = []
  for i in xrange(q_count):
    line = raw_input()
    arr = line.split(' ')
    q_id = int(arr[0])

    for t_id in arr[2:]:
      # binary searching an ordered list
      index = bisect.bisect_left(keys, int(t_id))
      topic = topic_list[index]
      topic.questions.append(q_id)
      # add this topic into the global dictionary as well
      # the dictionary is used to make sure no topic will 
      # be duplicated
      qtopic_dict[t_id] = topic
  # Build up another tree of questioned topics for searching
  # for topic in qtopic_dict.values():
  qtopicTree = KDTree.construct_from_data(qtopic_dict.values()) 
  
  # finally, reading the queries and outputing the results
  for i in xrange(query_count):
    line = raw_input()
    arr = line.split(' ')
    if arr[0]=='t':
      t_ids = topicTree.query([float(arr[2]), float(arr[3])], int(arr[1]))
      printArray(t_ids)

    elif arr[0]=='q':
      q_ids = qtopicTree.query([float(arr[2]), float(arr[3])], int(arr[1]), 
                                            KDTreeQuestionedNNs)
      printArray(q_ids)

if __name__=="__main__":
  readraw()
