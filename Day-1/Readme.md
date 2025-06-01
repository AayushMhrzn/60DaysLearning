# Day 01 – Functions and Classes

## Topics Covered

- Function definition and calling
- Class and objects
- Constructors (`__init__` method)
- Instance methods
- Class methods

## Explanation

### Functions
A function is a reusable block of code.


### Class

Classes and Objects
A class is a blueprint to create objects (instances). The constructor (__init__) initializes the object attributes.

Class Methods
Class methods are bound to the class, not the instance. They can modify class state that applies across all instances.

# BankAccount Class Example

## Description

This example demonstrates a simple `BankAccount` class with:

- **Constructor** to initialize the account holder's name and starting balance.
- **deposit(amount)** method to add money to the balance.
- **withdraw(amount)** method to remove money from the balance with checks for sufficient funds.
- A **class method** `show_accounts_created()` to display the total number of bank accounts created.

