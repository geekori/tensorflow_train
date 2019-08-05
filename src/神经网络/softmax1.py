import numpy as np
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
# [0.01690191 0.22756284 0.75553525]
# [0.01690191 0.22756284 0.75553525]
a = np.array([0.3,2.9,4.1])
print(softmax(a))

