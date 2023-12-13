import pandas as pd
import numpy as np


# missing_pct
missing_pct = 0.05
size = 500

f1 = np.random.rand(500, 1)
f2 = np.random.rand(500, 1)

indices = np.random.choice(np.arange(size), replace=False,
                           size=int(size * missing_pct))
f1[indices] = 0


indices = np.random.choice(np.arange(size), replace=False,
                           size=int(size * missing_pct))
f2[indices] = 0

arr = np.hstack((f1, f2))

df = pd.DataFrame(arr)
df.to_csv("data/test_dataset.csv", index=False)
