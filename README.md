# Puppy-Trainer-Prediction-using-SVM

The main goal of this project is to predict the training outcome with some given puppy and trainer info. Use your selected ML algorithm.

2.a Explain how your selected algorithm can be used to classify our problem. Comment on any possible drawback in using your selected algorithm for our prediction problem.

2.b Describe which Spark ML library can be used for prediction purpose with your selected algorithm.

2.c Apply Spark MLLib algorithm with puppy info alone on each set of data prepared in (1.b). List the prediction outcome vs. feature size in a table like the one shown below. Offer explanation on your observation of what feature sets appear to offer better prediction than others. (??%)

2.d Repeat each cell of table 2.c 10 times. Do you see consistent prediction rate for the same feature set of data when it is randomly split? How about when the data is not randomly split? Fill in each trial result and the average of trials in the table as shown follows. ( repeat for each of your 5 feature dataset as needed) (??%)

2.e What is the optimal number of features in your that offers best prediction rate based on 2.e? What features are in the dataset? Is it consistent with your 2.c result? (??%)

2.f List in a table the choice of parameters. e.g., maxdepth, choice of gradient descent vs. stochastic gradient descent, etc., that your algorithm provides. Check on the options you tried. Explain which parameters worked better than others. Highlight in bold the prediction rate and the parameters used for your best prediction trial. Indicate in your report what program(s) and data to run and how to run to get this best prediction rate. Provide a weblink to an online algorithm parameter choice overview that your parameter selection was based upon. (??%)

3.a For DayInLife column in TrainerInfo.xslt, normalize all the data within. See text normalization definition in Wiki (Links to an external site.)Links to an external site.. Show in a list of what normalization steps you did and with what code/library. Provide screenshots as appropriate. (??%)

3.b Perform feature extraction suitable for your prediction need. Explain what feature extraction in Spark MLlib (Links to an external site.)Links to an external site. did you use for this purpose. Show what feature extraction you did and describe what code/library was used. Provide screenshots as appropriate. (??%)

3.c Build and train your model based on the data in 3.b. Explain how you make use of your extracted features to build your model. Do you see consistent prediction rate when the data is randomly split? How about when the data is not randomly split? Fill in each trial result and the average of trials in the table as shown follows.  (??%)

-Bonus-
B.1 If you blend both the trainer and puppy info used in 2.e in building/training a new model, is the prediction rate better or worse? Why?

B.2 Do you see consistent prediction rate when the data is randomly split? How about when the data is not randomly split? Fill in each trial result and the average of trials in the table as shown follows.  (??%)
 
B.3 Select few columns of puppy info that is text based and in sentences. Perform normalization and feature extraction on these selected columns. Describe with screenshots and code how you did these.

B.4 Add few other selected columns with numeric data to these transformed text columns and build/train a model before applying it for your prediction. Do you see consistent prediction rate for the same feature set of data when it is randomly split? How about when the data is not randomly split? Fill in each trial result and the average of trials in the table as shown follows.

B.5 With the transformed trainer info and selected puppy info from both text and numeric columns, build/train a model before applying it for your prediction. Do you see consistent prediction rate for the same feature set of data when it is randomly split? How about when the data is not randomly split? Fill in each trial result and the average of trials in the table as shown follows.
