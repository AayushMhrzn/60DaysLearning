#function defining
def introduction(name):
    return f"My name is {name}."
#function calling
print (introduction("Aayush"))

#creating class
class Student:
    # Constructor method
    def __init__(self,name,year):
        self.name = name
        self.year = year
    # Instance method
    def introduce(self):
        return f"I am {self.name} studying in {self.year}year."
    
# Create an object of Student class
student1 = Student("Aayush","4th")
# Call instance method
print(student1.introduce())

# Class methods
class Circle:
    pi = 3.14159  # class attribute

    def __init__(self, radius):
        self.radius = radius

    # Instance method
    def area(self):
        return Circle.pi * (self.radius ** 2)

    # Class method - operates on the class, not instance
    @classmethod
    def change_pi(cls, new_pi):
        cls.pi = new_pi

circle1 = Circle(5)
print("Area with default pi:", circle1.area())

# Change class attribute pi using class method
Circle.change_pi(3.14)
print("Area after changing pi:", circle1.area())