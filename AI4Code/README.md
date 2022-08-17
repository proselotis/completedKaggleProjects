Note: I joined this project with about 10 days left to go. This compeition was similar to a pet project of mine, and rather than just read the resources, I felt it would be good to jump in fully and see how I could perform in the time I had in the competition. 

## Software Stack
Due to the complexity of this competition GPU resources seemed essential, however after the compeition a brillant submission with much less reliance was revealed to [place 21st on the public LB](https://www.kaggle.com/competitions/AI4Code/discussion/343614). I used Saturn cloud computing and various servers with varying cores and GPUs in order to optimize training. Alas these models tended to need around ~8 hours to get through one epoch so with my time disadvantage I struggled with this. 

## Competition Basics 
Given a python notebook, predict the order of the cells, using features from the code and limiting features from the data. The evaluation metric was a combination of how many swaps it takes to properly organize the notebook and how similar this was to the worst case scenario. 

## Data Basics 
All of the notebooks were pulled directly from Kaggle. To ensure that these notebooks are of interest, (from my understanding) the training data consisted of forked notebooks and forks of those forks. This caused [popular kagglers to have multiple enteries in the data](https://www.kaggle.com/competitions/AI4Code/discussion/324690). The test set was to be pulled from a 90 day window of no forked notebooks. It was also guaranteed that any kagglers who participated in the competition would not have their notebooks in the test set. 



### Joining so late and the heavy training time, I focused on learning what models were popular and how they were to be used rather than fine tuning my cross validation, feature engineering etc. 

## Common methods:
Word Embedding models: 
* [Codebert](https://huggingface.co/microsoft/codebert-base)
* [Graphcodebert](https://huggingface.co/microsoft/graphcodebert-base)
* [Codebertapy](https://huggingface.co/mrm8488/CodeBERTaPy) 
* [CodeT5](https://github.com/salesforce/CodeT5)


[My base model set](https://www.kaggle.com/competitions/AI4Code/discussion/343614): This method combined one of the code word embedding models above with a pairwise comparator.  
_Note: Unfortunately this Kaggler was fairly new to the platform, he released a combined version of a few notebooks (which is fine), but he included the weights of his model just a week before the end of the competition. This left [him](https://www.kaggle.com/dohuuthieu) just outside of medal range, but many people were able to copy this work and then he made it private again. I will be curious to see how Kaggle handles these copied datasets across teams._ 

When training my pairwise comparator, I found decent results but unfortunately it took to long to inference at run time so I wasn't able to use the model. My final submission were a combination of ensembles of the top 3 above code models with a layer added for ranking the location of each cell within the notebook. 



## TODO: Learning bits 


## TODO: Ending takeaway

