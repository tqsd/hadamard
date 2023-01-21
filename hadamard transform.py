from sympy import fwht

# sequence
seq = [1,1,1,1,1,1,1,1]

# hwht
transform = fwht(seq)
print("Transform  : ", transform)