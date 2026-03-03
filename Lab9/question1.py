import pandas as pd

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#t = 80
left_80 = data[data["BP"] <= 80]
right_80 = data[data["BP"] > 80]

print("t = 80")
print("Left size:", len(left_80))
print("Right size:", len(right_80))

#t = 78
left_78 = data[data["BP"] <= 78]
right_78 = data[data["BP"] > 78]

print("\nt = 78")
print("Left size:", len(left_78))
print("Right size:", len(right_78))

#t = 82
left_82 = data[data["BP"] <= 82]
right_82 = data[data["BP"] > 82]

print("\nt = 82")
print("Left size:", len(left_82))
print("Right size:", len(right_82))