
# Remove duplicates from a list

nums = [1, 2, 2, 3, 4, 4, 5]
nums = set(nums)
print(list(nums))

uniques = {}
result = []
for number in nums:
    uniques[number] = uniques.get(number, 0) + 1

for key in uniques.keys():
    result.append(key)

print(result)


# Count frequencies of words

words = ["apple", "banana", "apple", "orange", "banana", "apple"]

from collections import Counter

freq = Counter(words)

for key, val in freq.items():
    print(f"{key}: {val}")



# Find Duplicate Elements

nums = [1, 2, 3, 2, 4, 5, 3]

from collections import Counter

freq = Counter(nums)

result = []
for key, val in freq.items():
    if val > 1:
        result.append(key)

print(result)



# Separate Integers and Floats

data = [1, 2.5, 3, 4.7, 8]
integers = []
floats = []
for val in data:
    if type(val) == int:
        integers.append(val)
    elif type(val) == float:
        floats.append(val)
    else:
        pass

print(f"Integers: {integers}, Floats: {floats}")



# Group Students by Grade

students = [
    ("Arsh", "A"),
    ("Rahul", "B"),
    ("Priya", "A"),
]


result = {}
for name, grade in students:
    if grade not in result:
        result[grade] = []
    result[grade].append(name)

print(result)