# Supervised Machine Learning: Regression and Classification
Solved exercises from the Supervised Machine Learning: Regression and Classification course by Andrew NG

## Week 2
```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###  
    fwb = w*x + b
    cost = (fwb - y)**2
    total_cost = 1/(2*m)*sum(cost)
    ### END CODE HERE ### 

    return total_cost
```
