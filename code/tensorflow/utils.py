def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0: 
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    if size % 2 == 1: 
        median = data[(size-1)//2]
        data[0] = median
    return data[0]