# Code used to implement the math for the system

This section contains the code used to implement the mathematical side of the project. I will not go into the proof's of the mathematics used here since that was covered elsewhere.

## Code used to implement the tangent plane and circle point finder

```py title="Equation_solver_tangent_plane.py" linenums="1"

# importing from sympy library
from sympy import symbols, Eq, solve, I
import math
import time

def tangent_point_finder(x_0, y_0, Radius):
    # defining the symbolic variable 'z'
    start = time.perf_counter()
    x = symbols('x')

    # setting up the complex equation z^2 + 1 = 0
    equation = Eq(x**2*(x_0**2+y_0**2) - 2*Radius**2*x*x_0 + Radius**2*(Radius**2-y_0**2), 0)

    # solving the equation symbolically to find complex solutions
    solutions = solve(equation, x)
    if  x_0 > 0:
        x_t = solutions[0]

    else:
        x_t = solutions[1]

    y_t = math.sqrt(Radius**2 - x_t**2)
    end = time.perf_counter()
    inference_time = end - start
    print(f"inference time: {inference_time}")
    return [float(x_t), float(y_t)]

x = 30
y = 40
tangent_point_finder(x, y, 33.1)

```

The above code uses a library called sympy which helps in solving equations with algebraic representation.

## Closed form solution code

The code given below uses a closed form solution for finding the coordinates.

```py title="Equation_solver_closed_form.py" linenums="1"

from math import sqrt
import time
import random

def tangent_point_finder(x0, y0, R):
        start = time.perf_counter()
        denominator = x0**2 + y0**2
        if denominator == 0:
            raise ValueError("Denominator cannot be zero (x0 and y0 cannot both be zero).")
        
        term_inside_sqrt = R**4 * x0**2 - R**2 * (x0**2 + y0**2) * (R**2 - y0**2)
        
        if term_inside_sqrt < 0:
            raise ValueError("Negative value under the square root. No real solution.")
        
        sqrt_term = sqrt(term_inside_sqrt)
        
        x1 = (R**2 * x0 + sqrt_term) / denominator
        x2 = (R**2 * x0 - sqrt_term) / denominator

        if x0 > 0:
            x_t = x2

        if x0 < 0:
            x_t = x1

        y_t = sqrt(R**2 - x_t**2)

        end = time.perf_counter()
        inference_time = end - start
        print(f"inference time: {inference_time}")
        return [x_t, y_t]
```
Having a closed form solution make's the coordinate finding more faster.