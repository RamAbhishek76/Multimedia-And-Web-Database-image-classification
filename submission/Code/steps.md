### Feature Models and Feature represent the same meaning.


1) Finding the representative images for all the categories

2) Finding the representative images for all the categories and also for all the feature descriptors

3) task 0a - done, we already stored the categories (image labels)

4) task 0b - (image_id, feature_model, k); need to implement 1 & 2 for this. 
print the scores and feature space **bold**

5) task 1 - ("ying-yang", feature_model, k); need to implement 1 & 2 for this. 
print the scores and feature space **bold**

6) task 2a - (image_id, feature_model, k); 
print k matching labels, scores and feature space **bold**

7) task 2b - (image_id, k); implemented using RESNET
print k matching labels, scores **bold**

8) task 3 - (feature_model, k, dimensionality_reduction_technique)
Will need to create a properly labeled output file to store the top k latent semantics.
print { image_id:weight } dictionaries in the descending order of their weights.

9) task 4 - (feature_model, k)
Using the specified feature model, construct a three-modal tensor with dimensions corresponding to image, feature, and label.
Perform CP-decomposition on this tensor to extract latent semantics.
Will need to create a properly labeled output file to store the top k latent semantics.
print { image_id:weight } dictionaries in the descending order of their weights.