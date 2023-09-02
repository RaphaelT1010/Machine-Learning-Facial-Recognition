 # machinelearning-project

## Background
The dataset consists of 35,887 entries. Each entry has 3 columns: emotion, pixels, and usage.

Each face is classified into one of 7 emotions: anger, disgust, fear, happy, sad, surprise, or neutral. An encoding for these go into the emotion column for an entry.

The pixel information for each image goes into the pixel column (a 48x48 image, so 2304 values should be in each entry's pixels column. Values range from 0-255)

Lastly, usage decides whether or not an image will be used for training, publictest (validation), and privatetest. We will probably remove this.

The dataset contains a good variety of faces which range from varying ages, people with/without makeup, people of different races, actors, even some instances of virtual avatars, etc.

To summarize, the dataset contains a large variety and quantity of faces which will be very useful in making a machine learning model.

Link to database: https://ufile.io/40nmtjlw
## Introduction

The motivation for this project is to test the limit of our technical abilities by training an unsupervised model to assign its own emotion labels per grey scale image'
This method proves a unique challenge and insight into unsupervised learning models.

We want to train the first model to cluster images into groups seperated by emotion accurately, and a second model that is trained on those newly labeled images and can recognize human emotions from a test set.

The broader impact is giving anyone the ability to train a model to recognize human emotions without much human input.

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
