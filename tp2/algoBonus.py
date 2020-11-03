import pandas as pd
import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

labelencoder = preprocessing.LabelEncoder()

#Import dataset
gameDf= pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv",sep=',',encoding='iso-8859-1')

#Clean dataset
gameDf= gameDf.dropna()
gameDf= gameDf.drop(['Name', 'Year_of_Release','Genre','JP_Sales', 'Publisher', 'NA_Sales', 'EU_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Count', 'User_Count', 'Developer', 'Rating'],1)

#Encode dataset
gameDf.Platform = pd.Categorical(gameDf.Platform)
gameDf.Name = pd.Categorical(gameDf.Name)
gameDf.Platform = gameDf.Platform.cat.codes
gameDf.Name = gameDf.Name.cat.codes

# def convertUserScore(value):
#     value = float(value)*10
#     return float(value)
# gameDf.User_Score = gameDf.User_Score.apply(convertUserScore)


Y=gameDf['User_Score']
X=gameDf.drop('User_Score', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

randomforest = rf(n_estimators=10, max_features=3, max_depth=10)
randomforest.fit(X_train, Y_train)


#export model
joblib.dump(randomforest,'modelGame.pkl')