class BankAccount:
    # Class variable to keep track of number of accounts
    accounts_created = 0

    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance
        BankAccount.accounts_created += 1

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"{amount} deposited. New balance is {self.balance}.")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount <= 0:
            print("Withdrawal amount must be positive.")
        elif amount > self.balance:
            print("Insufficient balance.")
        else:
            self.balance -= amount
            print(f"{amount} withdrawn. New balance is {self.balance}.")

    @classmethod
    def show_accounts_created(cls):
        print(f"Total accounts created: {cls.accounts_created}")

# Example usage:
account1 = BankAccount("Aayush", 1000)
account1.deposit(500)
account1.withdraw(200)

account2 = BankAccount("Ram", 200)
account2.withdraw(300)  # Should warn insufficient balance

BankAccount.show_accounts_created()  # Output: Total accounts created: 2