#Artificial Neurons

There are two types of artificial neurons

 ### Perceptrons
 
 - Takes Binary Input
 - Returns Binary output

   0 = SUM of Wj * Xj  <= Threshold 
   1 = SUM of Wj * Xj  > Threshold 

   Threshold can be anything that is defined by the caller and is sometimes referred as bias or 'b'
   So   'b' ~~ -'Threshold'. 

   This can be taken up as a NAND function.

   0 =  W * X + b  <=  0
   1 = SUM of W * X + b  > 0

 ### Sigmoid

 - Takes any input b/w 0 & 1
 - Returns a sigmoid function
  
   Sig(z) = 1 / 1 + e.pow(-z)

 - <B>Why we started looking at Sigmoid</B>
 
   There  was a problem in Perceptrons: It was very difficult to change the values of W & B in a perceptron without affecting the whole network. 
   
   For e.g. Let's say we have a network that identifies digits & number '8' was identified incorrectly. 
   To fix this in a perceptron we  adjust the values of W & B and now for that case it identifies it correctly BUT the change in W & B now can & will cause the behaviour of the whole network as the out can only be binary.

- Even though we started by saying they are differnt, they are still alike, sigmoid is nothing but a smoothed out Step function (whereas a perceptron is a step function)


 
