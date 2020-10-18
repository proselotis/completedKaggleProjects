Links:
- [Competition](https://www.kaggle.com/c/landmark-recognition-2020/overview)
- [Leaderboard, finished 63th out of 736](https://www.kaggle.com/c/landmark-recognition-2020/leaderboard)



Things that worked:
- Tuning submission score only to ransac methods and not similarity measurements with all other photos
- Using Minkowski distance metric over cosine or others provided by sci-kit learn 
- Random sampling image classes to remove extra photos that describe same details from previous photos
- 


Things that didn't work:
- Image augmentation
  * random zoom
  * cropping 
  * color changes (gamma, contrast, brightness, etc.) 
- penalizing success with only one other photo
- Base resnet weights loaded in 
- weighing images based on a scale of unique photos 
- masking colors (RGB) 
- Implementing [DPDF](http://infolab.stanford.edu/~echang/dpf-ext.pdf), unsure if implemented incorrectly but inefficent to use within kaggle constraints
- Fine-tuning ransac methods and using other versions besides pydegensac (this was overfitting) 



