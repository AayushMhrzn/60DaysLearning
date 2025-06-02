from functools import reduce

# Lambda Function 
square = lambda x: x * x
print("Lambda square:", square(5))  # Output: 25

# Map Function 
nums = [1, 2, 3, 4, 5]
squared_nums = list(map(lambda x: x**2, nums))
print("Map -> squared numbers:", squared_nums)

# Reduce Function
product = reduce(lambda x, y: x * y, nums)
print("Reduce -> product of numbers:", product)

# Decorators and Wrappers 
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(">> Before function runs")
        result = func(*args, **kwargs)
        print(">> After function runs")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("Aayush")
