def ordinal_encoding(data):
    unique = []  #store unique categories

    for item in data:
        if item not in unique:
            unique.append(item)

    encoded = []

    for item in data:
        index = unique.index(item)  #position number
        encoded.append(index)

    return encoded

colors = ["Violet", "Indigo", "Blue", "Green", "Yellow", "Orange", "Red"]

result = ordinal_encoding(colors)
print("Ordinal Encoding:", result)

def one_hot_encoding(data):
    unique = []

    for item in data:
        if item not in unique:
            unique.append(item)

    encoded = []

    for item in data:
        row = []

        for category in unique:
            if item == category:
                row.append(1)
            else:
                row.append(0)

        encoded.append(row)

    return encoded

colors = ["Red", "Blue", "Green", "Blue", "Red"]

result = one_hot_encoding(colors)
print("One Hot Encoding:")
for r in result:
    print(r)