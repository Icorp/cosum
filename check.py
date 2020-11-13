import random
def randomizer_6(arr):
    result = []
    for i in range(len(arr)):
        sampling = random.sample(arr[i], 1)
        result.append(sampling)
    return result

lst = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]
a = randomizer_6(lst)
print(a)