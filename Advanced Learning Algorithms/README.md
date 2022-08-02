# Course 2: Advanced Learning Algorithms
Solved exercises from the course 2 Advanced Learning Algorithms course by DeepLearning.Ai / Andrew NG on Coursera

## Week 1

### Exercise 1

```python
model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense(25, activation='sigmoid'),
        Dense(15, activation='sigmoid'),
        Dense(1, activation='sigmoid')
        ### END CODE HERE ### 
    ], name = "my_model" 
)                            

```

### Exercise 2

```python
def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
### START CODE HERE ### 
    z = np.dot(a_in, W) + b
    a_out = g(z)   
### END CODE HERE ### 
    return(a_out)
```

### Exercise 3
```python
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
### START CODE HERE ### 
    z = np.matmul(A_in,W) + b
    A_out = g(z)
### END CODE HERE ### 
    return(A_out)
```
