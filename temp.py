import pandas as pd

conv_six = {0: 1, 1: 3, 2: 6, 3: 10, 4: 20, 5: 30}
X = [1, 2, 0, 5, 4]
y = list(map(lambda x: conv_six[x], X))
print(y)
