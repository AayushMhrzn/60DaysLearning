class Score:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Score({self.value})"

    def __add__(self, other):
        return Score(self.value + other.value)

    def __mul__(self, times):
        return Score(self.value * times)


class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting

    def __call__(self, name):
        print(f"{self.greeting}, {name}!")


def greet(name=None):
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello!")


# Test the features
s1 = Score(10)
s2 = Score(20)
print(s1 + s2)     # Score(30)
print(s1 * 3)      # Score(30)

g = Greeter("Hi")
g("Aayush")        # Hi, Aayush!

greet("Rita")      # Hello, Rita!
greet()            # Hello!
