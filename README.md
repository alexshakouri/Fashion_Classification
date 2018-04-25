# Fashion_Classification
This is from a challenge from kaggle.

the labels are not that hard to interpret. see https://www.kaggle.com/am1to2/data-exploration-and-analysis

procedure
1. data analysis

   a. map images to each label and try to interpret the learnability - DONE
   
   b. are there enough sample images for each label, if this is the case we can directly train a cnn with y= label, x = input. but it seems the dataset is not large enough. we need to try his first -YES
   
   c. plot the histograms of labels . 228  labels - DID
   d. 
   
   
2. Initial CNN training

   a. x_i - different sizes - resized the images- Is this the way? PADDING?
   
   b. y_i - multiple labels, how to feed them
   
      * binary label vector - TO BE TRIED
      
      * can we directly feed vectors of different lenghts
      
      * What else - GOOGLE
3. 
   
