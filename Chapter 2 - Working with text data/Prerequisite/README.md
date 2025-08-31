# Python Fundamentals for Machine Learning - Complete Study Guide

## Chapter 1: Control Flow Basics

### 1.1 Variables

**Concept**: Variables are labeled containers for storing information in memory.

**Key Features**:
- Python is dynamically typed - no need to declare types explicitly
- Types are inferred automatically
- Variables can change types during runtime
- Python is case-sensitive

**Examples**:

```python
# Basic variable assignment
x = 5                    # Integer
name = "Python"          # String
pi = 3.14               # Float
is_active = True        # Boolean

# Check variable types
print(type(x))          # <class 'int'>
print(type(name))       # <class 'str'>
print(type(pi))         # <class 'float'>
print(type(is_active))  # <class 'bool'>

# Dynamic typing - variables can change type
x = 5        # x is now an integer
x = "hello"  # x is now a string (no error)

# Case sensitivity
Variable = 10
variable = 20
print(Variable)  # 10
print(variable)  # 20 (different variables)
```

**Important Notes**:
- Use consistent naming conventions
- Variable discipline becomes crucial in larger ML projects
- Avoid changing variable types unnecessarily to prevent bugs

### 1.2 Data Structures

#### 1.2.1 Lists

**Concept**: Ordered, mutable collections that can store items of any type.

**Key Features**:
- Indexed starting from 0
- Support negative indexing (-1 for last element)
- Mutable - can be modified after creation
- Support slicing operations

**Examples**:

```python
# Creating lists
fruits = ["apple", "banana", "cherry", "date", "elderberry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Basic operations
print(len(fruits))        # 5 (length)
print(fruits[0])          # "apple" (first item)
print(fruits[-1])         # "elderberry" (last item)

# Slicing
print(fruits[1:3])        # ["banana", "cherry"] (items 1-2, 3 excluded)
print(fruits[:2])         # ["apple", "banana"] (first 2 items)
print(fruits[2:])         # ["cherry", "date", "elderberry"] (from index 2)

# Adding elements
fruits.append("fig")                    # Add to end
fruits.insert(1, "avocado")            # Insert at specific position

# Removing elements
fruits.remove("banana")                 # Remove specific item
del fruits[0]                          # Remove by index
popped = fruits.pop()                  # Remove and return last item

# Modifying elements
fruits[0] = "apricot"                  # Replace item at index 0

# Combining lists
exotic_fruits = ["mango", "papaya"]
all_fruits = fruits + exotic_fruits    # Using + operator
fruits.extend(exotic_fruits)           # Using extend method

# List of lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Aggregation functions for numeric lists
numbers = [1, 5, 3, 9, 2]
print(max(numbers))       # 9
print(min(numbers))       # 1
print(sum(numbers))       # 20
numbers.sort()           # [1, 2, 3, 5, 9]
```

#### 1.2.2 Tuples

**Concept**: Ordered, immutable collections ideal for fixed data sets.

**Key Features**:
- Created with parentheses ()
- Immutable - cannot be changed after creation
- Support indexing and slicing like lists
- Used for coordinates, fixed collections

**Examples**:

```python
# Creating tuples
coordinates = (3.5, 4.2)
rgb_color = (255, 128, 0)
student_info = ("Alice", 20, "Computer Science")

# Accessing elements
print(coordinates[0])     # 3.5
print(rgb_color[-1])      # 0 (last element)

# Attempting to modify raises an error
# coordinates[0] = 5.0    # This would cause an error

# Extending tuples (creates new tuple)
new_coordinates = coordinates + (1.0,)  # (3.5, 4.2, 1.0)

# Unpacking tuples
x, y = coordinates
name, age, major = student_info
```

#### 1.2.3 Dictionaries

**Concept**: Key-value pairs for fast lookups and data organization.

**Key Features**:
- Created with curly braces {}
- Keys must be unique and immutable
- Values can be of any type
- Excellent for structured data representation

**Examples**:

```python
# Creating dictionaries
student = {
    "name": "Alice",
    "age": 22,
    "major": "AI/ML"
}

# Accessing values
print(student["name"])        # "Alice"
print(student.get("age"))     # 22

# Adding/modifying entries
student["grade"] = "A"        # Add new key-value pair
student["age"] = 23           # Modify existing value

# Removing entries
del student["major"]          # Remove using del
grade = student.pop("grade")  # Remove and return value

# Dictionary methods
print(student.keys())         # Dict keys
print(student.values())       # Dict values
print(student.items())        # Key-value pairs
```

### 1.3 Conditionals

**Concept**: Decision-making structures that execute code based on conditions.

**Key Components**:
- `if` statement for primary condition
- `elif` for additional conditions
- `else` for fallback behavior
- Indentation defines code blocks

**Examples**:

```python
# Basic if-else
temperature = 25
if temperature > 25:
    print("It's hot outside")
else:
    print("It's cool outside")

# Multiple conditions with elif
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

# Logical operators
age = 20
has_permission = True

if age >= 18 and has_permission:
    print("Access granted")
elif age >= 18 and not has_permission:
    print("Need permission")
else:
    print("Too young")

# Truthiness in Python
empty_list = []
if empty_list:           # Empty list is falsy
    print("Has items")
else:
    print("List is empty")

# Falsy values: 0, None, "", [], {}, False
if 0:                    # Always false
    print("This won't print")

if None:                 # Always false  
    print("This won't print either")
```

**Important Notes**:
- Colon (:) is required after condition statements
- Indentation (usually 4 spaces) defines code blocks
- Multiple conditions evaluated top to bottom
- First true condition executes, others are skipped

### 1.4 Loops

#### 1.4.1 For Loops

**Concept**: Iterate through sequences or execute code a specific number of times.

**Examples**:

```python
# Basic for loop with range
for i in range(5):           # 0, 1, 2, 3, 4
    print(f"Count: {i}")

# Range with start, stop, step
for i in range(1, 10, 2):    # 1, 3, 5, 7, 9
    print(i)

# Iterating over lists
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# Iterating over dictionaries
student = {"name": "Alice", "age": 22, "major": "CS"}
for key in student:
    print(f"{key}: {student[key]}")

# Or iterate over key-value pairs
for key, value in student.items():
    print(f"{key}: {value}")

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

#### 1.4.2 While Loops

**Concept**: Continue execution while a condition remains true.

**Examples**:

```python
# Basic while loop
count = 5
while count > 0:
    print(f"Countdown: {count}")
    count -= 1                    # Must update condition variable

# While loop with user input
user_input = ""
while user_input.lower() != "quit":
    user_input = input("Enter command (or 'quit'): ")
    if user_input.lower() != "quit":
        print(f"You entered: {user_input}")

# Infinite loop protection
max_iterations = 100
count = 0
while True:
    count += 1
    if count >= max_iterations:
        break
    # Some processing here
```

#### 1.4.3 Loop Control

**Concept**: Modify loop behavior with `break` and `continue`.

**Examples**:

```python
# Using continue to skip iterations
for i in range(5):
    if i == 2:
        continue    # Skip when i is 2
    print(i)        # Prints: 0, 1, 3, 4

# Using break to exit loop early
for i in range(10):
    if i == 3:
        break       # Exit loop when i is 3
    print(i)        # Prints: 0, 1, 2

# Practical example: finding first even number
numbers = [1, 3, 7, 8, 9, 12, 15]
for num in numbers:
    if num % 2 == 0:
        print(f"First even number: {num}")
        break
```

## Chapter 2: Functions and Classes

### 2.1 Functions

**Concept**: Reusable blocks of code that perform specific tasks.

**Structure**:
```python
def function_name(parameters):
    """Optional docstring"""
    # Function body
    return value  # Optional
```

**Examples**:

```python
# Basic function
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Function with multiple parameters
def calculate_total_price(price, quantity, tax_rate=0.1):
    """Calculate total price with tax"""
    subtotal = price * quantity
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total

# Function with no parameters
def get_greeting():
    """Return a greeting message"""
    return "Hello, welcome to ML!"

# Function returning multiple values
def analyze_numbers(numbers):
    """Return statistics for a list of numbers"""
    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    return total, average, maximum

# Function with no return (performs action)
def print_summary(name, score):
    """Print a formatted summary"""
    print(f"Student: {name}")
    print(f"Score: {score}")
    print("-" * 20)

# Usage examples
temp_f = celsius_to_fahrenheit(25)
total, avg, max_val = analyze_numbers([1, 2, 3, 4, 5])
print_summary("Alice", 95)
```

#### 2.1.1 Special Function Types

**Lambda Functions**:
```python
# Anonymous functions for simple operations
square = lambda x: x ** 2
print(square(5))  # 25

# Useful for sorting and transformations
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
students.sort(key=lambda student: student[1])  # Sort by score
```

**Recursive Functions**:
```python
def factorial(n):
    """Calculate factorial recursively"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

**Higher-Order Functions**:
```python
def apply_operation(numbers, operation):
    """Apply an operation to each number"""
    return [operation(num) for num in numbers]

def double(x):
    return x * 2

def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
doubled = apply_operation(numbers, double)
squared = apply_operation(numbers, square)
```

**Generator Functions**:
```python
def fibonacci_generator(n):
    """Generate Fibonacci sequence up to n terms"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Usage
fib_gen = fibonacci_generator(10)
for num in fib_gen:
    print(num, end=" ")  # 0 1 1 2 3 5 8 13 21 34
```

### 2.2 Classes and Objects

**Concept**: Classes are blueprints for creating objects that bundle data and behavior.

**Basic Structure**:
```python
class ClassName:
    def __init__(self, parameters):
        # Constructor - initializes object
        self.attribute = value
    
    def method_name(self, parameters):
        # Method - function that belongs to the class
        # Use self to access object attributes
        return value
```

**Examples**:

```python
# Basic class
class Adder:
    def __init__(self):
        self.last_result = None
    
    def add(self, a, b):
        result = a + b
        self.last_result = result
        return result
    
    def get_last_result(self):
        return self.last_result

# Creating and using objects
my_adder = Adder()
result = my_adder.add(5, 3)        # 8
last = my_adder.get_last_result()  # 8

# Multiple objects maintain separate state
adder1 = Adder()
adder2 = Adder()
adder1.add(10, 20)    # adder1.last_result = 30
adder2.add(5, 7)      # adder2.last_result = 12
```

**More Complex Example**:
```python
class Student:
    # Class variable (shared by all instances)
    school_name = "ML Academy"
    
    def __init__(self, name, age, major):
        # Instance variables (unique to each object)
        self.name = name
        self.age = age
        self.major = major
        self.grades = []
    
    def add_grade(self, grade):
        """Add a grade to the student's record"""
        self.grades.append(grade)
    
    def get_average(self):
        """Calculate and return average grade"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def get_info(self):
        """Return formatted student information"""
        avg = self.get_average()
        return f"{self.name}, {self.age} years old, {self.major} major, Average: {avg:.2f}"
    
    @classmethod
    def get_school_name(cls):
        """Class method to access class variables"""
        return cls.school_name

# Usage
alice = Student("Alice", 20, "Computer Science")
alice.add_grade(85)
alice.add_grade(92)
alice.add_grade(78)

print(alice.get_info())
print(f"School: {Student.get_school_name()}")
```

#### 2.2.1 Object-Oriented Programming Concepts

**Encapsulation**:
```python
class BankAccount:
    def __init__(self, initial_balance=0):
        self._balance = initial_balance  # Protected attribute
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            return amount
        return 0
    
    def get_balance(self):
        return self._balance

# Usage
account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # 1500
```

**Inheritance**:
```python
# Base class
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

# Derived classes
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

class Cow(Animal):
    def speak(self):
        return f"{self.name} says Moo!"

# Usage
animals = [Dog("Rex"), Cat("Whiskers"), Cow("Bessie")]
for animal in animals:
    print(animal.speak())
```

**Enhanced Adder with Inheritance**:
```python
class FancyAdder(Adder):
    def multiply(self, a, b):
        result = a * b
        self.last_result = result
        return result
    
    def get_operation_count(self):
        # This is just for demonstration
        return "Operations performed"

# Usage
fancy = FancyAdder()
fancy.add(5, 3)        # Inherited method
fancy.multiply(4, 7)   # New method
```

### 2.3 When to Use Functions vs Classes

**Use Functions When**:
- You need to perform a specific task
- Logic is stateless
- Simple input-output operations
- Utility functions

**Use Classes When**:
- You need to represent something with both data and behavior
- Managing state is important
- Building complex systems
- Modeling real-world entities

## Chapter 3: Vectors - The Heart of Machine Learning

### 3.1 Understanding Vectors

**Concept**: Collections of numbers that work together to describe something meaningful.

**Real-World Examples**:
- **Location**: (latitude, longitude) = (51.5074, -0.1278) for London
- **Shopping list**: (3 bananas, 5 apples, 2 mangoes) = [3, 5, 2]  
- **RGB Color**: (red, green, blue) = (255, 128, 0) for orange
- **Student features**: (age, GPA, study_hours) = [20, 3.8, 25]

**Geometric Interpretation**:
- Vector as an arrow from origin to point (3, 4)
- Has magnitude (length) and direction
- Magnitude = √(3² + 4²) = 5
- Direction = angle from positive x-axis

### 3.2 Vectors in Machine Learning

**Why Vectors Matter**:
- All ML data must be converted to numbers
- Vectors preserve structure and relationships
- Enable mathematical operations
- Foundation for all ML computations

**ML Applications**:
- **Images**: 28×28 image becomes 784-dimensional vector
- **Text**: Words/sentences become embedding vectors
- **Structured Data**: Each row becomes a feature vector
- **Audio**: Sound waves become time-series vectors
- **Model Parameters**: Weights and biases stored as vectors

### 3.3 Creating Vectors with NumPy

**Basic Vector Operations**:

```python
import numpy as np

# Creating vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

print(f"Vector 1: {vec1}")
print(f"Type: {type(vec1)}")
print(f"Shape: {vec1.shape}")

# Vector arithmetic
addition = vec1 + vec2        # [5, 7, 9]
subtraction = vec2 - vec1     # [3, 3, 3]
scalar_mult = vec1 * 3        # [3, 6, 9]
dot_product = np.dot(vec1, vec2)  # 32

# Aggregation functions
print(f"Sum: {np.sum(vec1)}")           # 6
print(f"Mean: {np.mean(vec1)}")         # 2.0
print(f"Standard deviation: {np.std(vec1)}")

# Quick creation methods
zeros = np.zeros(5)              # [0, 0, 0, 0, 0]
ones = np.ones(3)                # [1, 1, 1]
range_vec = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
random_vec = np.random.rand(4)   # Random values between 0 and 1
```

**Comparison: Lists vs NumPy Arrays**:

```python
# Python lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list_sum = list1 + list2        # [1, 2, 3, 4, 5, 6] (concatenation)

# NumPy arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr_sum = arr1 + arr2           # [5, 7, 9] (element-wise addition)
```

### 3.4 Creating Vectors with PyTorch

**Why PyTorch for ML**:
- Full deep learning framework
- GPU acceleration
- Automatic differentiation
- Seamless integration with neural networks

**Basic PyTorch Tensors**:

```python
import torch

# Creating tensors (1D = vectors)
vec1 = torch.tensor([1.0, 2.0, 3.0])
vec2 = torch.tensor([4, 5, 6])

print(f"Vector 1: {vec1}")
print(f"Type: {type(vec1)}")
print(f"Shape: {vec1.shape}")

# Similar operations as NumPy
addition = vec1 + vec2
dot_product = torch.dot(vec1, vec2.float())

# Quick creation methods
zeros = torch.zeros(5)
ones = torch.ones(3)
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)
random_vec = torch.rand(4)
normal_vec = torch.randn(4)  # From normal distribution

# Moving to GPU (if available)
if torch.cuda.is_available():
    vec_gpu = vec1.cuda()
    print(f"Device: {vec_gpu.device}")

# For Apple Silicon Macs
if torch.backends.mps.is_available():
    vec_mps = vec1.to('mps')
    print(f"Device: {vec_mps.device}")
```

## Chapter 4: PyTorch Tensors

### 4.1 Understanding Tensors

**Concept**: Generalization of vectors and matrices to any number of dimensions.

**Tensor Hierarchy**:
- **Scalar (0D)**: Single number
- **Vector (1D)**: Array of numbers
- **Matrix (2D)**: Grid of numbers (rows × columns)
- **3D Tensor**: Cube of numbers
- **Higher dimensions**: For complex data structures

**Dimensional Examples**:
```python
import torch

# Scalar (0D tensor)
scalar = torch.tensor(42)
print(f"Scalar shape: {scalar.shape}")  # torch.Size([])

# Vector (1D tensor)  
vector = torch.tensor([1, 2, 3, 4])
print(f"Vector shape: {vector.shape}")  # torch.Size([4])

# Matrix (2D tensor)
matrix = torch.tensor([[1, 2, 3], 
                       [4, 5, 6]])
print(f"Matrix shape: {matrix.shape}")  # torch.Size([2, 3])

# 3D tensor
tensor_3d = torch.tensor([[[1, 2, 3], 
                           [4, 5, 6]], 
                          [[7, 8, 9], 
                           [10, 11, 12]]])
print(f"3D tensor shape: {tensor_3d.shape}")  # torch.Size([2, 2, 3])
```

### 4.2 Creating Tensors

**From Raw Data**:

```python
import torch

# From Python lists
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

# Explicit data types
float_tensor = torch.tensor([1.0, 2.0, 3.0])  # float32 by default
int_tensor = torch.tensor([1, 2, 3])          # int64 by default
bool_tensor = torch.tensor([True, False, True])

# Specifying data type explicitly
specific_type = torch.tensor([1, 2, 3], dtype=torch.float32)
```

**Using Built-in Functions**:

```python
# Zeros and ones
zeros_2d = torch.zeros(3, 4)        # 3×4 matrix of zeros
ones_3d = torch.ones(2, 3, 4)       # 2×3×4 tensor of ones
full_tensor = torch.full((2, 3), 7)  # 2×3 matrix filled with 7

# Identity matrix
identity = torch.eye(3)              # 3×3 identity matrix

# Random tensors
random_uniform = torch.rand(2, 3)    # Uniform distribution [0, 1)
random_normal = torch.randn(2, 3)    # Standard normal distribution
random_int = torch.randint(0, 10, (2, 3))  # Random integers

# From ranges
arange_tensor = torch.arange(0, 10)         # [0, 1, 2, ..., 9]
linspace_tensor = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### 4.3 Tensor Inspection

**Essential Attributes**:

```python
# Create a sample tensor
tensor = torch.randn(2, 3, 4)

# Shape and dimensionality
print(f"Shape: {tensor.shape}")      # torch.Size([2, 3, 4])
print(f"Size: {tensor.size()}")      # Same as shape
print(f"Dimensions: {tensor.ndim}")  # 3
print(f"Number of elements: {tensor.numel()}")  # 24

# Data type and device
print(f"Data type: {tensor.dtype}")  # torch.float32
print(f"Device: {tensor.device}")    # cpu

# Memory layout
print(f"Requires grad: {tensor.requires_grad}")  # False by default
```

**Device Management**:

```python
# Check device and move tensors
tensor_cpu = torch.tensor([1, 2, 3])
print(f"Original device: {tensor_cpu.device}")

# Move to GPU (NVIDIA)
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.cuda()
    # or
    tensor_gpu = tensor_cpu.to('cuda')
    print(f"GPU device: {tensor_gpu.device}")

# Move to Apple Silicon GPU
if torch.backends.mps.is_available():
    tensor_mps = tensor_cpu.to('mps')
    print(f"MPS device: {tensor_mps.device}")

# Move back to CPU
tensor_back = tensor_gpu.cpu()
```

### 4.4 Data Type Conversions

```python
# Converting between PyTorch tensors and other formats
tensor = torch.tensor([1, 2, 3, 4])

# Tensor to Python scalar (only for single-element tensors)
scalar_tensor = torch.tensor(42)
python_value = scalar_tensor.item()

# Tensor to NumPy array (shares memory!)
numpy_array = tensor.numpy()
print(f"NumPy array: {numpy_array}")

# NumPy to tensor (also shares memory!)
import numpy as np
np_array = np.array([5, 6, 7, 8])
tensor_from_np = torch.from_numpy(np_array)

# Warning: Shared memory means changes affect both!
numpy_array[0] = 999
print(f"Original tensor: {tensor}")  # First element changed to 999!
```

### 4.5 Best Practices and Common Pitfalls

**Recommended Practices**:

```python
# 1. Use torch.tensor() (lowercase 't')
good_tensor = torch.tensor([1, 2, 3])    # Recommended
# avoid: torch.Tensor([1, 2, 3])        # Less safe

# 2. Always verify shapes before operations
def safe_operation(a, b):
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor b shape: {b.shape}")
    if a.shape == b.shape:
        return a + b
    else:
        raise ValueError("Shape mismatch!")

# 3. Explicit data type specification
tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

# 4. Device consistency
def ensure_same_device(tensor1, tensor2):
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    return tensor1, tensor2
```

**Common Mistakes to Avoid**:

```python
# 1. Device mismatch error
tensor_cpu = torch.tensor([1, 2, 3])
if torch.cuda.is_available():
    tensor_gpu = torch.tensor([4, 5, 6]).cuda()
    # This will cause an error:
    # result = tensor_cpu + tensor_gpu  # Device mismatch!
    
    # Correct way:
    result = tensor_cpu.cuda() + tensor_gpu

# 2. Shape mismatch
a = torch.tensor([[1, 2]])      # Shape: [1, 2]
b = torch.tensor([[1], [2]])    # Shape: [2, 1]
# result = a + b  # This will broadcast, might not be intended

# 3. Data type issues
int_tensor = torch.tensor([1, 2, 3])
float_tensor = torch.tensor([1.0, 2.0, 3.0])
# Mixed operations might cause unexpected type promotion
result = int_tensor + float_tensor  # Result will be float
```

## Chapter 5: Tensor Manipulation

### 5.1 Indexing and Slicing

**Basic Indexing**:

```python
import torch

# Create sample tensors
matrix = torch.tensor([[1, 2, 3], 
                       [4, 5, 6]])

tensor_3d = torch.tensor([[[1, 2, 3], 
                           [4, 5, 6]], 
                          [[7, 8, 9], 
                           [10, 11, 12]]])

# Single element access
print(matrix[0, 1])        # Element at row 0, column 1: 2
print(matrix[1, -1])       # Last element of second row: 6

# Row and column access
print(matrix[0])           # First row: tensor([1, 2, 3])
print(matrix[:, 1])        # Second column: tensor([2, 5])

# Multiple elements
print(matrix[0, :2])       # First row, first 2 columns: tensor([1, 2])
print(matrix[1:, 1:])      # From second row, from second column
```

**Advanced Slicing**:

```python
# Step slicing
large_tensor = torch.arange(20).reshape(4, 5)
print(large_tensor)
print(large_tensor[::2])           # Every second row
print(large_tensor[:, ::2])        # Every second column
print(large_tensor[::2, ::2])      # Every second row and column

# Negative indexing
print(large_tensor[-1])            # Last row
print(large_tensor[:, -2:])        # Last two columns

# Ellipsis (...) for high-dimensional tensors
tensor_4d = torch.randn(2, 3, 4, 5)
print(tensor_4d[..., 0].shape)     # All dimensions except last, select first element
print(tensor_4d[0, ..., -1].shape) # First batch, all middle dims, last element
```

**Boolean Indexing (Masking)**:

```python
# Create sample data
data = torch.tensor([1, -2, 3, -4, 5, -6, 7])

# Create boolean mask
positive_mask = data > 0
print(f"Mask: {positive_mask}")     # tensor([True, False, True, False, True, False, True])

# Apply mask to filter data
positive_values = data[positive_mask]
print(f"Positive values: {positive_values}")  # tensor([1, 3, 5, 7])

# Direct conditional indexing
even_values = data[data % 2 == 0]
print(f"Even values: {even_values}")

# Complex conditions
large_positive = data[(data > 0) & (data > 3)]
print(f"Large positive: {large_positive}")

# Mask with different tensor (must match dimensions)
scores = torch.tensor([85, 76, 92, 68, 88, 79, 95])
high_scores = scores[scores >= 80]
print(f"High scores: {high_scores}")

# Common mistake: dimension mismatch
try:
    wrong_mask = torch.tensor([True, False])  # Only 2 elements
    filtered = data[wrong_mask]  # data has 7 elements - will error
except:
    print("Dimension mismatch error!")
```

### 5.2 Reshaping and View Operations

**Basic Reshaping**:

```python
# Start with a flat tensor
original = torch.arange(12)
print(f"Original: {original}")          # tensor([0, 1, 2, ..., 11])
print(f"Original shape: {original.shape}")  # torch.Size([12])

# Reshape to 2D matrix
matrix = original.reshape(3, 4)
print(f"Reshaped to matrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape}")   # torch.Size([3, 4])

# Reshape to 3D tensor
tensor_3d = original.reshape(2, 2, 3)
print(f"3D tensor shape: {tensor_3d.shape}")  # torch.Size([2, 2, 3])

# Using view() - similar to reshape but with stricter memory requirements
view_matrix = original.view(4, 3)
print(f"View matrix:\n{view_matrix}")

# Flatten to 1D using -1
flattened = matrix.reshape(-1)           # or matrix.view(-1)
print(f"Flattened: {flattened}")
print(f"Flattened shape: {flattened.shape}")  # torch.Size([12])

# Let PyTorch infer one dimension
auto_reshape = original.reshape(3, -1)   # 3 rows, infer columns (will be 4)
print(f"Auto reshape shape: {auto_reshape.shape}")  # torch.Size([3, 4])
```

**Reshape vs View**:

```python
# Create sample tensor
data = torch.arange(6)

# reshape() is more flexible
reshaped = data.reshape(2, 3)

# view() requires contiguous memory layout
viewed = data.view(2, 3)

# Both share memory with original
reshaped[0, 0] = 999
print(f"Original after reshape change: {data}")  # First element changed

# Demonstrate the difference with non-contiguous tensors
non_contiguous = data.t()  # Transpose makes it non-contiguous
try:
    # This will work
    safe_reshape = non_contiguous.reshape(3, 2)
    print("Reshape worked on non-contiguous tensor")
    
    # This might fail
    risky_view = non_contiguous.view(3, 2)
    print("View also worked (tensor was made contiguous)")
except:
    print("View failed on non-contiguous tensor")

# Common error: incompatible dimensions
try:
    bad_reshape = original.reshape(5, 3)  # 12 elements can't fit in 5x3 (15 spots)
except:
    print("Error: Total elements must remain the same")
```

### 5.3 Adding and Removing Dimensions

**Unsqueeze - Adding Dimensions**:

```python
# Start with 1D vector
vector = torch.tensor([1, 2, 3])
print(f"Original shape: {vector.shape}")  # torch.Size([3])

# Add dimension at different positions
# unsqueeze(0) adds dimension at position 0
row_vector = vector.unsqueeze(0)
print(f"Row vector shape: {row_vector.shape}")    # torch.Size([1, 3])
print(f"Row vector:\n{row_vector}")

# unsqueeze(1) adds dimension at position 1
col_vector = vector.unsqueeze(1)
print(f"Column vector shape: {col_vector.shape}")  # torch.Size([3, 1])
print(f"Column vector:\n{col_vector}")

# Can also use negative indices
alt_col = vector.unsqueeze(-1)  # Same as unsqueeze(1) for 1D tensor
print(f"Alt column shape: {alt_col.shape}")

# Multiple unsqueezes
multi_dim = vector.unsqueeze(0).unsqueeze(-1)
print(f"Multi-dimensional shape: {multi_dim.shape}")  # torch.Size([1, 3, 1])

# Alternative syntax with None or slice notation
also_row = vector[None, :]      # Same as unsqueeze(0)
also_col = vector[:, None]      # Same as unsqueeze(1)
```

**Squeeze - Removing Dimensions**:

```python
# Start with tensor that has dimensions of size 1
padded = torch.tensor([[[1], [2], [3]]])
print(f"Padded shape: {padded.shape}")    # torch.Size([1, 3, 1])

# Remove all dimensions of size 1
squeezed = padded.squeeze()
print(f"Squeezed shape: {squeezed.shape}")  # torch.Size([3])

# Remove specific dimension
squeeze_first = padded.squeeze(0)          # Remove first dimension
print(f"Squeeze first: {squeeze_first.shape}")  # torch.Size([3, 1])

squeeze_last = padded.squeeze(-1)          # Remove last dimension
print(f"Squeeze last: {squeeze_last.shape}")   # torch.Size([1, 3])

# squeeze() only removes dimensions of size 1
normal_tensor = torch.randn(2, 3, 4)
still_same = normal_tensor.squeeze()
print(f"No change in shape: {still_same.shape}")  # torch.Size([2, 3, 4])
```

**Practical Examples**:

```python
# Converting vector for matrix operations
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Need to reshape vector for matrix multiplication
col_vector = vector.unsqueeze(1)      # Make it [3, 1]
result = torch.matmul(matrix.float(), col_vector)
print(f"Matrix multiplication result shape: {result.shape}")

# Batch processing - adding batch dimension
single_image = torch.randn(3, 28, 28)    # Single RGB image
batch = single_image.unsqueeze(0)        # Add batch dimension
print(f"Single image: {single_image.shape}")  # torch.Size([3, 28, 28])
print(f"Batched: {batch.shape}")              # torch.Size([1, 3, 28, 28])
```

### 5.4 Transposing and Permuting

**Matrix Transpose**:

```python
# 2D matrix transpose
matrix = torch.tensor([[1, 2, 3], 
                       [4, 5, 6]])
print(f"Original matrix:\n{matrix}")
print(f"Original shape: {matrix.shape}")    # torch.Size([2, 3])

# Transpose using .t() attribute
transposed = matrix.t()
print(f"Transposed matrix:\n{transposed}")
print(f"Transposed shape: {transposed.shape}")  # torch.Size([3, 2])

# Alternative: .T property (similar to NumPy)
also_transposed = matrix.T
print(f"Also transposed shape: {also_transposed.shape}")
```

**Multi-dimensional Transpose**:

```python
# For tensors with more than 2 dimensions, use transpose()
tensor_3d = torch.randn(2, 3, 4)
print(f"Original 3D shape: {tensor_3d.shape}")

# Swap specific dimensions
swapped = tensor_3d.transpose(0, 2)  # Swap dimensions 0 and 2
print(f"Swapped (0,2): {swapped.shape}")     # torch.Size([4, 3, 2])

swapped_12 = tensor_3d.transpose(1, 2)  # Swap dimensions 1 and 2
print(f"Swapped (1,2): {swapped_12.shape}")  # torch.Size([2, 4, 3])

# Common use case: image tensor format conversion
# From [Batch, Channels, Height, Width] to [Batch, Height, Width, Channels]
image_batch = torch.randn(10, 3, 224, 224)  # 10 RGB images, 224x224
hwc_format = image_batch.transpose(1, 3).transpose(1, 2)
print(f"BCHW to BHWC: {image_batch.shape} -> {hwc_format.shape}")
```

**Permute - Reordering All Dimensions**:

```python
# Start with 4D tensor (batch, channels, height, width)
tensor_4d = torch.randn(8, 3, 64, 64)
print(f"Original BCHW: {tensor_4d.shape}")

# Reorder to (batch, height, width, channels)
bhwc = tensor_4d.permute(0, 2, 3, 1)
print(f"Permuted to BHWC: {bhwc.shape}")

# Another reordering: (height, width, batch, channels)
hwbc = tensor_4d.permute(2, 3, 0, 1)
print(f"Permuted to HWBC: {hwbc.shape}")

# Permute can achieve any dimension ordering
custom_order = tensor_4d.permute(3, 1, 0, 2)
print(f"Custom order: {custom_order.shape}")

# Practical example: preparing data for different ML frameworks
# PyTorch typically uses BCHW, TensorFlow typically uses BHWC
def pytorch_to_tensorflow_format(tensor):
    """Convert PyTorch BCHW to TensorFlow BHWC"""
    return tensor.permute(0, 2, 3, 1)

def tensorflow_to_pytorch_format(tensor):
    """Convert TensorFlow BHWC to PyTorch BCHW"""
    return tensor.permute(0, 3, 1, 2)
```

### 5.5 Expanding and Repeating

**Expand - Memory-Efficient Broadcasting**:

```python
# Start with a column vector
col_vector = torch.tensor([[1], [2], [3]])
print(f"Column vector shape: {col_vector.shape}")  # torch.Size([3, 1])

# Expand to larger size without copying data
expanded = col_vector.expand(3, 4)
print(f"Expanded shape: {expanded.shape}")         # torch.Size([3, 4])
print(f"Expanded tensor:\n{expanded}")

# Expand is memory efficient - data is not actually copied
print(f"Original tensor:\n{col_vector}")
col_vector[0, 0] = 999
print(f"After changing original:\n{expanded}")  # All first row elements changed!

# Expand with -1 to keep original size
tensor_2d = torch.tensor([[1, 2]])              # Shape: [1, 2]
expanded_rows = tensor_2d.expand(5, -1)         # Shape: [5, 2]
print(f"Expanded rows:\n{expanded_rows}")

# Can expand multiple dimensions
single_element = torch.tensor([[42]])           # Shape: [1, 1]
big_tensor = single_element.expand(3, 4)       # Shape: [3, 4]
print(f"Expanded single element:\n{big_tensor}")
```

**Repeat - Physical Data Copying**:

```python
# Start with same column vector
col_vector = torch.tensor([[1], [2], [3]])

# Repeat creates actual copies
repeated = col_vector.repeat(1, 4)  # Repeat 1 time along dim 0, 4 times along dim 1
print(f"Repeated shape: {repeated.shape}")     # torch.Size([3, 4])
print(f"Repeated tensor:\n{repeated}")

# Changes to original don't affect repeated tensor
original_value = col_vector[0, 0].item()
col_vector[0, 0] = 888
print(f"Original changed to: {col_vector[0, 0]}")
print(f"Repeated first element still: {repeated[0, 0]}")  # Unchanged

# Repeat along multiple dimensions
small = torch.tensor([[1, 2]])
big_repeat = small.repeat(3, 2)  # 3 times vertically, 2 times horizontally
print(f"Big repeat:\n{big_repeat}")

# Repeat vs tile (tile is an alias for repeat)
tiled = torch.tile(small, (3, 2))
print(f"Tiled (same as repeat):\n{tiled}")
```

**When to Use Expand vs Repeat**:

```python
# Use expand when:
# 1. You need shape compatibility for broadcasting
# 2. Memory efficiency is important
# 3. You won't modify the expanded tensor

def efficient_broadcasting_example():
    vector = torch.tensor([1, 2, 3, 4]).unsqueeze(0)  # [1, 4]
    matrix = torch.randn(5, 4)
    
    # Method 1: Expand for broadcasting (memory efficient)
    expanded_vector = vector.expand(5, -1)
    result1 = matrix + expanded_vector
    
    # Method 2: Direct broadcasting (even more efficient)
    result2 = matrix + vector  # PyTorch handles broadcasting automatically
    
    return result1, result2

# Use repeat when:
# 1. You need independent copies
# 2. You plan to modify the result
# 3. The operation requires actual data duplication

def independent_copies_example():
    base_weights = torch.tensor([0.1, 0.2, 0.3])
    
    # Create multiple independent copies for different experiments
    experiment_weights = base_weights.repeat(5, 1)  # 5 independent copies
    
    # Modify each experiment independently
    for i in range(5):
        experiment_weights[i] *= (i + 1)  # Different scaling for each
    
    return experiment_weights

# Memory usage comparison
def memory_comparison():
    base = torch.randn(1000, 1)
    
    # Expand: minimal memory usage
    expanded = base.expand(1000, 1000)
    print(f"Base memory: {base.element_size() * base.nelement()} bytes")
    print(f"Expanded shares memory with base")
    
    # Repeat: full memory allocation
    repeated = base.repeat(1, 1000)
    print(f"Repeated memory: {repeated.element_size() * repeated.nelement()} bytes")
```

### 5.6 Advanced Indexing Examples

**Practical ML Scenarios**:

```python
# Scenario 1: Batch processing with variable sequence lengths
def pad_sequences_example():
    # Different length sequences
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    seq3 = torch.tensor([6, 7, 8, 9])
    
    # Find max length
    max_len = max(len(seq1), len(seq2), len(seq3))
    
    # Pad sequences
    batch = torch.zeros(3, max_len, dtype=torch.long)
    batch[0, :len(seq1)] = seq1
    batch[1, :len(seq2)] = seq2
    batch[2, :len(seq3)] = seq3
    
    print(f"Padded batch:\n{batch}")
    return batch

# Scenario 2: Selecting specific samples from a batch
def batch_selection_example():
    # Batch of images: [batch_size, channels, height, width]
    batch_images = torch.randn(10, 3, 32, 32)
    
    # Select specific images (e.g., indices 0, 3, 7)
    selected_indices = torch.tensor([0, 3, 7])
    selected_images = batch_images[selected_indices]
    print(f"Selected images shape: {selected_images.shape}")
    
    # Advanced indexing: select different channels for each image
    batch_size = batch_images.shape[0]
    # Select channel 0 for first 3 images, channel 1 for next 3, channel 2 for rest
    channel_indices = torch.tensor([0]*3 + [1]*3 + [2]*4)
    image_indices = torch.arange(batch_size)
    
    selected_channels = batch_images[image_indices, channel_indices]
    print(f"Selected channels shape: {selected_channels.shape}")

# Scenario 3: Masking invalid data
def masking_invalid_data():
    # Temperature readings with some invalid values
    temperatures = torch.tensor([23.5, -999, 25.1, 22.8, -999, 24.0])
    
    # Mask invalid readings (assuming -999 indicates invalid)
    valid_mask = temperatures != -999
    valid_temperatures = temperatures[valid_mask]
    print(f"Valid temperatures: {valid_temperatures}")
    
    # Replace invalid values with mean of valid ones
    mean_temp = valid_temperatures.mean()
    temperatures_cleaned = temperatures.clone()
    temperatures_cleaned[~valid_mask] = mean_temp
    print(f"Cleaned temperatures: {temperatures_cleaned}")

# Run examples
pad_sequences_example()
batch_selection_example()
masking_invalid_data()
```

### 5.7 Common Pitfalls and Debugging

**Shape Debugging Strategies**:

```python
def debug_tensor_shapes(*tensors, operation_name="operation"):
    """Utility function to debug tensor shapes"""
    print(f"\n--- Debugging {operation_name} ---")
    for i, tensor in enumerate(tensors):
        print(f"Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")
    print("-" * 40)

# Common mistake 1: Broadcasting mismatch
def broadcasting_mistake():
    try:
        a = torch.randn(3, 4)
        b = torch.randn(2, 4)  # Different first dimension
        debug_tensor_shapes(a, b, operation_name="broadcasting")
        # result = a + b  # Will error - shapes don't broadcast
    except Exception as e:
        print(f"Broadcasting error: {e}")

# Common mistake 2: Dimension order confusion
def dimension_confusion():
    # Image data in different formats
    pytorch_format = torch.randn(1, 3, 224, 224)  # BCHW
    tensorflow_format = torch.randn(1, 224, 224, 3)  # BHWC
    
    debug_tensor_shapes(pytorch_format, tensorflow_format, 
                       operation_name="format mismatch")
    
    # Wrong way to combine them
    # combined = torch.cat([pytorch_format, tensorflow_format])  # Will error
    
    # Correct way: convert to same format first
    tf_to_pytorch = tensorflow_format.permute(0, 3, 1, 2)
    combined = torch.cat([pytorch_format, tf_to_pytorch])
    print(f"Successfully combined: {combined.shape}")

# Common mistake 3: Squeeze/unsqueeze confusion
def squeeze_unsqueeze_mistakes():
    # Starting with image batch
    images = torch.randn(1, 3, 64, 64)  # Single image in batch
    
    # Mistake: squeezing without specifying dimension
    squeezed_all = images.squeeze()  # Removes batch dimension too!
    debug_tensor_shapes(images, squeezed_all, 
                       operation_name="squeeze all")
    
    # Correct: specify which dimension to squeeze
    squeezed_batch = images.squeeze(0)  # Only remove batch dimension
    debug_tensor_shapes(images, squeezed_batch, 
                       operation_name="squeeze batch only")

# Run debugging examples
broadcasting_mistake()
dimension_confusion()
squeeze_unsqueeze_mistakes()
```

## Summary and Key Takeaways

### Essential Operations Checklist:

```python
# Essential tensor operations you should master:

# 1. Shape inspection
def inspect_tensor(tensor, name="tensor"):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

# 2. Indexing and slicing
tensor = torch.randn(4, 5, 6)
first_batch = tensor[0]           # Get first batch
subset = tensor[1:3, :, 2:5]      # Slice multiple dimensions
masked = tensor[tensor > 0]       # Boolean indexing

# 3. Reshaping
flattened = tensor.view(-1)       # Flatten to 1D
reshaped = tensor.reshape(2, 2, 15)  # Reshape to compatible dimensions

# 4. Dimension manipulation  
expanded = tensor.unsqueeze(1)    # Add dimension
squeezed = expanded.squeeze(1)    # Remove dimension

# 5. Transposing and permuting
transposed = tensor.transpose(0, 1)     # Swap two dimensions
reordered = tensor.permute(2, 0, 1)     # Reorder all dimensions

# 6. Expanding and repeating
broadcasted = tensor[:1].expand(4, -1, -1)  # Memory-efficient expansion
copied = tensor[:1].repeat(4, 1, 1)         # Physical copying
```

This completes the comprehensive guide to tensor manipulation in PyTorch. These operations form the foundation for all machine learning operations, from data preprocessing to model building and training.
