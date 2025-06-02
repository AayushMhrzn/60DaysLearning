class Employee:
    """Base class for all employees."""

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_details(self):
        return f"Name: {self.name}, Salary: {self.salary}"

    def work(self):
        return f"{self.name} is doing general employee work."


class Developer(Employee):
    """Developer class inherits from Employee."""

    def __init__(self, name, salary, programming_language):
        super().__init__(name, salary)  # Call parent constructor
        self.programming_language = programming_language

    def get_details(self):
        base = super().get_details()
        return f"{base}, Programming Language: {self.programming_language}"

    def work(self):
        return f"{self.name} is coding in {self.programming_language}."


class Manager(Employee):
    """Manager class inherits from Employee."""

    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size

    def get_details(self):
        base = super().get_details()
        return f"{base}, Team Size: {self.team_size}"

    def work(self):
        return f"{self.name} is managing a team of {self.team_size} members."


# Testing the classes
dev = Developer("Aayush", 80000, "Python")
mgr = Manager("Rita", 100000, 5)

print(dev.get_details())    # Detailed info of Developer
print(dev.work())           # Custom work method

print(mgr.get_details())    # Detailed info of Manager
print(mgr.work())           # Custom work method
