b = ["foo", "bar", "baz", "bar"]
indices = [i for i, x in enumerate(b) if x == "bar"]
print(indices)