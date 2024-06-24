import numpy as np
import pandas as pd
import taichi as ti

ti.init(arch=ti.cpu)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
arr1 = ti.field(dtype=ti.f32, shape=(3, 2))
arr1.from_numpy(df.to_numpy())

lst_1 = df['a', 'b'].to_numpy()
lst_2 = df['b'].to_numpy()

print(lst_1, lst_2)
