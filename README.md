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

```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ### 
    fwb = w*x + b
    dj_db = (1/m)*sum(fwb - y)
    dj_dw = (1/m)*sum((fwb - y)*x)
    ### END CODE HERE ### 
        
    return dj_dw, dj_db
```
    

