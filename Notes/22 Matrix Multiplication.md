# Matrix Multiplication

- How to check two matrices can be multiplied
  - Check the shape of matrices [e.g. Shape 1 - (4, 2), Shape 2 - (2, 3) ] 
  - put the shapes together like 
    4, 2 , 2, 3
  - If the inner shape has same value (2, 2 are same)
  - They can be multiplied.  
- Order of multiplication, can alter multiplicity
- i.e 
  - Shape 1 * Shape 2 => Allowed)
  - Shape 2 * Shape 1 = NOT Allowed)


- Output of multiplication

MartrixA = [ 
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8]
           ]   
                
MartrixB = [ 
            [10, 30, 50],
            [20, 40, 60]                   
           ]  

  - Shape of output will be what is left out of the shape
  - Out put shape will be [4, 3]

- How it is done
        
    matrixc [0][0] = 
        matrixA[0][0] * matrixcB[0][0] +
        matrixA[0][1] * matrixcB[1][0] + 
    
    matrixC [1][0] = 
        matrixA[1][0] * matrixcB[0][0] +
        matrixA[1][1] * matrixcB[1][0] + 


