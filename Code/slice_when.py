def slice_when(predicate, iterable):
    i, x, size = 0, 0, len(iterable)
    while i < size-1:
        if predicate(iterable[i], iterable[i+1]):
            if len(iterable[x:i+1])>2: yield iterable[x:i+1]
            x = i + 1
        i += 1
    yield iterable[x:size]