# machinelearning-project

## Introduction

We aim to create an unsupervised model and cluster the images via k-means clustering (while also applying PCA). This means taking out the emotion encodings/column (at least) of the original dataset. 

![Kmeans Clustering](images/KmeansClustering.png)

We hope that the optimal number of clusters is 7 (like the original dataset, the number of emotions), which we then will manually assign each cluster an emotion label based on the closest images to each centroid. We then will assign the corresponding emotion to every image in that cluster and compare it with the original images in the dataset to see if the emotions are correct.

Creating a model for facial recognition is really cool to think about because a machine identifying faces and emotions was something unheard of in the early age of computers. Creating a somewhat successful model as a group of college students would be an achievement.

We wanted to test our technical abilities as well by choosing an unsupervised, image-based project rather than a supervised one based on continuous values. This would allow us to push our limits and give us more confidence for more difficult projects in the future.

The broader impact is that we could give anyone the ability to use our model and classify emotions with any image dataset they have. They would be able to train a model to recognize human emotions without much human input (just by uploading their image dataset)

## Methodology
### Data Exploration
Our dataset is named fer2013.csv and is a dataset that consists of 35,887, 28x28, pixelated images. As such, the dataset consists of 35,887 entries. Each entry has 3 columns: emotion, pixels, and usage. 

Each face is classified into one of 7 emotions: anger, disgust, fear, happy, sad, surprise, or neutral. An encoding for these goes into the emotion column for an entry. The pixel information for each image goes into the pixel column (a 48x48 image, so 2304 values should be in each entry's pixels column. Values range from 0-255). Usage is either training, public test, or private test, which is similar to saying which images should be used for training, validation, or testing.

The dataset contains a wide variety of faces, including individuals of varying ages, people with and without makeup, individuals from different racial backgrounds, actors, and even some instances of virtual avatars. In summary, this dataset offers a substantial and diverse collection of faces, making it highly valuable for training machine learning models.

The dataset consisted of 35887 images/entries. We found this through calling the len() function on the dataset. We found that each entry had 3 columns: emotion, pixels, and usage and found it through calling .columns on the dataset.

We confirmed that the dataset has seven encodings for each emotion by calling df[‘emotion’].unique(). We also confirmed that each pixel entry had 2304 different numbers, as our 48x48 image should have this much. We also confirmed that there were three different usages by calling df[‘Usage’].unique(),

We found that the emotion categories had a significant imbalance.  We did this through calling df.emotion.value_counts() and found that emotion 3 (happy) had significantly more images than the emotion 1(disgust), with 8989 and 546 images respectively.

There were no null/nan values as well, which we confirmed with .isna(). We called .dropna() for good measure afterwards.

Lastly, we plotted images from each emotion category to ensure there were no deformities and that each image was appropriately labeled. We created a function called displayClasses and basically plotted two images from each emotion category. We scanned our dataset for the first two occurrences, and implemented plot/subplot to get all images into one plot.

Link to database: https://ufile.io/40nmtjlw

### Data Preprocessing
We found that the dataset was mostly preprocessed already.  The images have been compressed to be 48 x 48 and have already been converted to greyscale. 

Because our dataset was quite large (at around 36k images), we decided to cut down on it. We decided to get 547 images from each of the seven emotions, totalling at 3829 images for our model to train and test on. 

As a result, we created a script that could get x images from each emotion category (we could set x to anything in case we felt like we needed more images). Our cut down dataset would be contained in a csv called ‘face-emo.csv’ which we then would normalize.

We also normalized all pixel values. This was done using min max scalar.

We also dropped the emotion encoding and usage columns of the dataset. 

### Models

## Model 1 Ver. 1
Our first model would be focused around K-means clustering and seeing if we should use either an SVD or PCA approach.

To test the quality of clustering in machine learning, silhouette scores are often calculated. We decided to test the silhouette scores of both an SVD and PCA approach. They were very similar, with PCA being slightly better, so we clustered according to PCA.

We then wanted to check the optimal number of components and clusters for our dataset after our initial clustering with PCA. We set a range for the number of principal components (2-5) and the number of clusters (5-15) to see if our model could potentially spot other emotions. Any combination that would increase our silhouette score would be the one that we would use, with us using cross validation on top of this. 

In the end, everything gave a roughly similar silhouette score (around .33) so we decided to go with our initial 2 principal components, 7 cluster plot.

We then plotted random images from each cluster and see if any cluster followed an inherent pattern or similar emotion. We then plotted the 14 images closest to each cluster centroid to see if the emotions shown would be similar. Ideally, those closest to the centroid would show similar emotions, as their principal components should be the same.

We then used the plots of the above images and manually mapped a cluster back to a certain emotion. This was based on what we saw the most. We then applied an emotion label back to all images based on which cluster they belonged to.

We then called accuracy_score to compare how our model clustered each image according to PCA (with our manual labels) with the original dataset’s emotion labels. Any mismatch would lead our accuracy to be lower and any match would lead our accuracy to be higher.

## Results
#### DUMP OUT THE RESULTS HERE BUT DONT TALK ABOUT THEM, ADD DIAGRAMS OF RESULTS, THE CLUSTERING DIAGRAM, OR OTHERS THAT SHOW OUR RESULTS
#### LAST PARAGRAPH HERE IS ABOUT THE FINAL MODEL AND FINAL RESULTS SUMMARY

## Discussion
#### TALK ABOUT THE RESULTS HERE, AND OUR THOUGHT PROCESS BEGINNING TO END. REALLY THINK IN THIS SECTION AND SHOW HOW YOU THINK SCIENTIFICALLY 
In the preprocessing step, we decided to only grab 547 sample images from each class of emotion because one of the classes of emotion only had 547 sample images. Initially, we wanted more than 547 samples to train the model but having more than 547 samples will lead to bias in our model. Therefore, we only have 547 samples from each emotion class. The data was already downsized and gray-scaled so there wasn’t any other thing we would have done other than normalizing the pixels values of each image and dropping the columns “emotion” and “usage”. The column “emotion" is the classification of each image and the “usage” columns specify whether the image will be used for testing or training. We don't need these columns because we are doing unsupervised ML to classify each image and doing our own splitting of the images for testing and training. Before we do any machine learning, we wanted to check and remove any null data so it won’t impact the results of our model. 

Our first model was not that great. The images closest to the centroids of each cluster seem to have no relationship to each other in human eyes but there is some patterns and relationship that only a computer can see. This is to say that there exists some underlying relationship between each image that could be used to better classify the images. We tried printing out more and more of each image at the centroids of each cluster but it seems like every image’s emotion is pretty much random to the naked eyes even though they are images from the centroid of each cluster and should have some relationship to each other. This could mean that the computer may have came out with different kind of classification that we are unaware of. For example, emotions like angry, sad, or happy is obvious to humans but to computer, it is entirely different. Therefore, we concluded that these emotions classification is something only computer can understand what it means and impossible for humans to understand these classification. Out of curiosity, we try a larger set of data to see if it yields better accuracy. And it did! But it only improved the accuracy by a little bit. 

Since we are going for an unsupervised approach we don't really have a true over/underfitting situation so we attempted to compare the clusters to the original dataset. We mapped out our clustered images to the emotions from the original dataset. This was done by plotting a random set of images from each cluster and then mapping out each emotion visually to their respective original emotion. This process was challenging because the clustering wasn't great so we mapped to the emotion that was most dominant. We got an accuracy of about 14% meaning that there was little correlation to the original emotions. Our next steps are to improve the model to increase our silhouette score, which in turn might increase the classification accuracy of the model.

## Conclusion
#### WRAP UP, MIND DUMP, OPINIONS FUTURE PLANS, WHAT WOULD WE DO DIFFERENT.

## Collaboration
#### EVERYONE WILL BE FILL THIS PART OUT FOLLOW EXAMPLE...
### Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.
### Dillon Jackson: Wroked on the readme, accuracy testing, and project facilitation 
### Billy Ouattara: Wrote the script for balancing the different type of images, and built the initial unsupervised model. 
