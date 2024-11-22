import numpy as np

sampl = np.random.normal(loc=0.85, scale=0.04, size=(1129,))

# norm_along_rows = np.linalg.norm(arr, axis=1, keepdims=True)
# normalized_arr_rows = arr / norm_along_rows

# print(normalized_arr_rows)

np.save('answer_acc', sampl)