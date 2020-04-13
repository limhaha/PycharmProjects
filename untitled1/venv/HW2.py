k = 'Korean', 'Mathematics', 'English'

v = 90.3, 85.5, 92.7


def make_dict(k,v):
    D = dict(zip(k,v))
    return D

D = make_dict(k,v)

for i in k:
    for j in v:
        while j == D[i]:
            print(i, "&", j, " -> Correct")
            break
