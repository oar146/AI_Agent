import numpy as np
#  你只需要知道计算余弦相似度会用到一个函数，
#  你输入两段文本，函数会返回给你它们的相似度分数。
#  也就是说，函数只是个工具，你会用这个工具就行，无需知道它们的原理
def get_dot(vec_a, vec_b):
    """计算两个向量的点积，两个向量同维度数字乘积之和"""
    if len(vec_a) != len(vec_b):
        raise ValueError('两个向量维度不一致,维度相同才能计算点积')
    dot_sum = 0
    for a,b in zip(vec_a, vec_b):
        dot_sum += a * b
    return dot_sum
# 这个其实就是在做向量a*向量b 那个余弦公式的分子


def get_norm(vec):
    """计算单个向量的模长"""
    sum_square = 0
    for i in vec:
        sum_square += i ** i
    return np.sqrt(sum_square)



def get_cos(vec_a, vec_b):
    """计算两个向量的余弦相似度"""
    dot_product = get_dot(vec_a, vec_b)
    norm_a = get_norm(vec_a)
    norm_b = get_norm(vec_b)
    cos = dot_product / (norm_a * norm_b)
    return cos
if __name__ == '__main__':
    vec_a = [1, 2, 3]
    vec_b = [2, 3, 4]
    print(get_cos(vec_a, vec_b))