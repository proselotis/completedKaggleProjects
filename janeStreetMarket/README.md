Since this competition's results are ongoing until August 21 2021 I will not be discussing my successes or model strategy until after then. 

## Links:
- [Competition](https://www.kaggle.com/c/jane-street-market-prediction/overview)
- [Leaderboard](https://www.kaggle.com/c/jane-street-market-prediction/leaderboard)



## Things that worked:
- Discussion of these to come when the competition ends (August 21 2021) 


## Things that didn't work:
- Dimensionality reduction of the response variables to help simplify the model (scores did not favor this)
- Dimensionality reduction of the variable groupings (found in the features.csv, features that were created using other features) to simplify the input of the model (time to run the code was a problem
- Optimizer tuning appropriately (Optimizers with momentum worked in specific situations of training the data, but I did not feel the results were consistent enought to include them within my final models) 

## Ideas that didn't get implemented:
- [Feature neutrlization](https://www.kaggle.com/code1110/janestreet-avoid-overfit-feature-neutralization): I played around with this idea but couldn't get it into my code functionally and with the time contrainsts of the project. 

## Things that people implemented to overfit:
- Removing all days before date 85 
- Removing all high frequency trading days
- Removing all data with no weight 



## Overall blurb
Very fun competition that surprisingly did not allow time series comparisons (no indicator of what opportunity was for the same stock/portfolio/chance to act) some of the key problems are discussed below: 
1. The data has a significantly different trends after the dates marked passed day 85, people found this and omitted the days, potentially overfitting the data. There was a lot of discussion of when the data was created and people speculated that day 85 was when covid hit but no confirmation from the hosts. 
2. The competition has it's own consistent utility score, each opportunity has a weight associated with it based on how much it could help you. Users found removing any row with weight less than zero for models would improve their score (anticipated overfitting). These models with optimized public scores had a high chance of overfitting decision thresholds and model parameters to what worked within this subsetted dataset. In my models I began to minimze the difference between removing those weighted rows and keeping them in and found small differences by the end of my work and thus decided it would not be useful to submit a potentially overfitted file. 
3. The response variable had 5 different responses, it was up to the user to decide if those results were good and you should choose to act on the opportunity or not. I did not see much investigation into these values from the public but I looked into this to maximize results. However each time you change the response your results on models is now on a new scale and it becomes harder to compare your local CV changes thus this can slow you down. 
4. The constant battle of what appeared to be overfitting from the public and discovering what insightful strategies people were using, some people advocated for having a model that constantly retrains itself during the results phase and others voted for more complex models based on the data that was given and used their alloted time to submit predictions through such models. 

Overall there were some cool ideas discussed, and a user did a good job of summing them up in a post [here](https://www.kaggle.com/c/jane-street-market-prediction/discussion/221495)
