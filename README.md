# NFL-Big-Data-Bowl-2026

**Team note as of 11/3:**
FOCUS ONLY ON baseRmultivariate.R file

We need to think more carefully about variables putting into the model
1 player, 1 play no interaction
(can't have more covariates than data points)
see which 5 covariates are the highest
(derivative in placement and location)
s, a, v, direction

2 players, 1 play and see their interaction
3,4,5 ....

Once we can do this in 1 play, this will fix our errors in 1 game and so forth

## File Descriptions


 ==============================================================================
[tyreek hill link](https://www.statmuse.com/nfl/ask/tyreek-hill-2023)
Team project for predictive analytics. 

Project links and deadlines: 

[nfl-big-data-bowl-2026-analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics) December 17 submission deadline

[nfl-big-data-bowl-2026-prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) December 3 submission deadline

 ## Ideas of relationships to predict player movement
- type of play
- most common formations
- top routes
- interaction of location on the field and score to where they will be
- zone coverage is more than man coverage so add that to the model with defensive players
- based off of position and O/D it would help (then orientation and direction would help us know where offense will go since defense is more predictable with coverage)

## Vision of professors 
- for predictive competition, dr page want to throw a neural net at the data first and see what we come up with then feature engineer (edit interactions and explanatory variables) from there to predict better
- i ran a random forest but after 11 hours of running i quit it, so i'll try something else lol

## What we need done now
- fully explore the data (EDA and play by play animations for specific week or team to understand player movement details in relation to our variables)
  - focus on coverage for defense, offense, and position
  - include geospatial sphere for predictions?
- spend time thinking and researching of the prompt (what have people already done? what has been found? what is known and not known?)
- look at current submissions for analytics and predictions competition and see what they are doing
- **tracking data** = think differently about statistics that you have ever before

Look at these links for help on visualization and animation: 

[EDA + MODEL + VIDEO + 3D VIDEO](https://www.kaggle.com/code/taylorsamarel/eda-model-video-3d-video) 

[ Player Tracking Animation V1](https://www.kaggle.com/code/dedquoc/nfl-bdb-2026-player-tracking-animation-v1)

[how guy link below made his animations](https://www.kaggle.com/code/mohammedshammeer/package-for-animating-tracking-visualization)

[NFL Animations made simple | nfl-tracks](https://www.kaggle.com/code/mohammedshammeer/nfl-animations-made-simple-nfl-tracks)
