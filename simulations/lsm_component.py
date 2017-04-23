class LSMComponent():
  def __init__(self, size):
    self.hits = 0
    self.misses = 0
    self.size = size
    self.entries = []

  @property
  def full(self):
    return len(self.entries) == self.size

  @property
  def empty(self):
    return len(self.entries) == 0

  @property
  def free_space(self):
    return self.size - len(self.entries)

  @property
  def accesses(self):
    return self.hits + self.misses

  @property
  def hit_frequency(self):
    return self.hits / self.accesses

  @property
  def miss_frequency(self):
    return self.misses / self.accesses