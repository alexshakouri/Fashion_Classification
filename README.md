# Fashion_Classification
This is from a challenge from kaggle.

the labels are not that hard to interpret. see https://www.kaggle.com/am1to2/data-exploration-and-analysis

procedure
1. data analysis
   \na. map images to each label and try to interpret the learnability
   \nb. are there enough sample images for each label, if this is the case we can directly train a cnn with y= label, x = input. but it seems the dataset is not large enough. we need to try his first
   \nc. plot the histograms of labels . 228  labels
   d. 