# machinelearning-project

## Background
Our dataset is named fer2013.csv and is a dataset that consists of 35,887, 28x28, pixelated images. As such, the dataset consists of 35,887 entries. Each entry has 3 columns: emotion, pixels, and usage. 

Each face is classified into one of 7 emotions: anger, disgust, fear, happy, sad, surprise, or neutral. An encoding for these goes into the emotion column for an entry. The pixel information for each image goes into the pixel column (a 48x48 image, so 2304 values should be in each entry's pixels column. Values range from 0-255). Usage is either training, public test, or private test, which is similar to saying which images should be used for training, validation, or testing.

The dataset contains a wide variety of faces, including individuals of varying ages, people with and without makeup, individuals from different racial backgrounds, actors, and even some instances of virtual avatars. In summary, this dataset offers a substantial and diverse collection of faces, making it highly valuable for training machine learning models.

Link to database: https://ufile.io/40nmtjlw
## Introduction

We aim to create an unsupervised model and cluster the images via k-means clustering (while also applying PCA). This means taking out the emotion encodings/column (at least) of the original dataset. 

We hope that the optimal number of clusters is 7 (like the original dataset, the number of emotions), which we then will manually assign each cluster an emotion label based on the closest images to each centroid. We then will assign the corresponding emotion to every image in that cluster and compare it with the original images in the dataset to see if the emotions are correct.

Creating a model for facial recognition is really cool to think about because a machine identifying faces and emotions was something unheard of in the early age of computers. Creating a somewhat successful model as a group of college students would be an achievement.

We wanted to test our technical abilities as well by choosing an unsupervised, image-based project rather than a supervised one based on continuous values. This would allow us to push our limits and give us more confidence for more difficult projects in the future.

The broader impact is that we could give anyone the ability to use our model and classify emotions with any image dataset they have. They would be able to train a model to recognize human emotions without much human input (just by uploading their image dataset)

## Methodology
### Data Preprocessing
The dataset is already almost completely preprocessed. The pixels have been compressed to be 48 x 48 and have already been converted to greyscale. This means we have to do little preprocessing on the images.

However, we plan to normalize the grey scale values within each entry's pixel column. This will make the pixel column more readable, and less greyscale intensity with numerous images may help with overall speed as normalizing may help with gradient calculation.

We also plan to get an even amount of images from each emotion class, 547, in order to ensure our model doesn't become bias to seeing one emotion too much and hopefully speed up computations for our dataset. This will also justify our removal for the usage column.

The images don't need to be cropped or resized. We don't need to crop them because the images are already pretty decently focused on people's face, and a good facial recognition model should be able to detect faces even in bad circumstances. They don't need to be resized too because they are already very small. Normalization will mostly likely occur. 

### Models
For our Face emotion recognition model, we decided to used unsupervised learning. We used kmeans clustering to cluster the data categorizing it into different groups that could represent the different type of emotions. In our model, the images are converted into 2D arrays. We used PCA to reduce the dimension of the images before fitting them into our kmeans model. Since we have a total of 7 type of emotions, we chose a cluster value of 7. To evaluate the efficiency of the model, we used the silhouette score as metric. So far when training the model we obtain a silhouette score of about 0.3. To improve the model, we applied cross validation. However, the score only increased by about 0.01. An alternative to improve our model could be data transformation or a different form of data preprocessing using an algorithm different from PCA.

### First Analysis
Since we are going for an unsupervised approach we don't really have a true over/under fitting situation so we attempted to compare the clusters to the original dataset. We mapped out our clustered images to the emotions from the original dataset. This was done by plotting a random set of images from each cluster and then mapping out each emotion visually to their respective original emotion. This process was challenging because the clustering wasn't great so we mapped to the emotion that was most dominant. We got an accuracy of about 14% meaning that there was little correlation to the original emotions. Our next steps are to improve the model to increase our silhouette score, which in turn might increase the classification accuracy of the model.

## Results
### Graphs
Describe some of the graphs we generate and what they mean

### Other Results (Rename later when we have results)
Talk about our error results and ant numerical results here.

## Conclusion
Wrap up the findings and information. \
Any future follow ups/investigations or future plans with the project?
