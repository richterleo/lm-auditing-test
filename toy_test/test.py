import pandas as pd

data = []

for i in range(5):
    a = {}

    a["a"] = i
    if i % 2 == 0:
        a["b"] = 2 * i

    data.append(a)

df = pd.DataFrame(data)
print(df)
