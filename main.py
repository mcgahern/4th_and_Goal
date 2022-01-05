import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing and Inspecting Data
data = pd.read_csv('play_by_play_2020.csv')

all_plays_2020 = data[['game_id','posteam', 'defteam', 'yardline_100',
                        'down', 'ydstogo', 'ydsnet', 'play_type', 'yards_gained',
                        'pass_length', 'pass_location', 'air_yards','first_down_rush',
                        'first_down_pass', 'incomplete_pass', 'interception','rush_attempt',
                        'pass_attempt', 'touchdown', 'passer_player_id','passer_player_name',
                        'receiver_player_id','receiver_player_name', 'receiving_yards',
                        'rusher_player_id', 'rusher_player_name', 'rushing_yards', 'series_success',
                        'series_result', 'play_type_nfl']]

# Filtering Data to just Green Bay
GB_plays = all_plays_2020['posteam'] == 'GB'
GB_2020 = all_plays_2020[GB_plays]

# Removing plays that went for > 18 yards
GB_2020 = GB_2020[GB_2020['yards_gained'].between(0, 18)]
# Removing plays that were under 8 yards needed and more than 18
GB_2020 = GB_2020[GB_2020['ydstogo'].between(8, 18)]

# Removing plays that are not passes or runs
GB_2020 = GB_2020[(GB_2020['play_type'] == 'pass') | (GB_2020['play_type'] == 'run')]
print(GB_2020.head(10))
print(GB_2020.shape)

# Passes under 8 yards won't be enough, so they will be given a 0 and > 8 will be given a 1
GB_2020['receiving_success'] = np.where(GB_2020['receiving_yards'] >= 8, 1, 0)
# Runs under 8 yards won't be enough, so they will be given a 0 and > 8 will be given a 1
GB_2020['rushing_success'] = np.where(GB_2020['rushing_yards'] >= 8, 1, 0)
# Where either one is a successful value
GB_2020['successful_play'] = np.where((GB_2020['rushing_success'] == 1) | (GB_2020['receiving_success'] == 1),1,0)
# Setting the plays into integer values
GB_2020['play_type_int'] = np.where(GB_2020['play_type'] == 'pass',1,0)
# Setting all other teams to 0 and the Buccs to 1
GB_2020['defense'] = np.where(GB_2020['defteam'] == 'TB', 1, 0)
# Changing GB rushers to integers
GB_2020['rusher_player_id'] = GB_2020['rusher_player_id'].fillna(0)
GB_2020["rusher_player_id"].replace({"00-0033293": 1, "00-0033948": 2,
                                     "00-0036265": 3, "00-0023459": 4}, inplace=True)
#Dropoping rushers who did not have a play vs Buccs
values = ["00-0032404", "00-0035161", "00-0034272", "00-0034995", "00-0035480"]
GB_2020 = GB_2020[GB_2020.rusher_player_id.isin(values) == False]
print(GB_2020[['rusher_player_name','rusher_player_id']].value_counts())

# Changing GB receivers to integers for players who received a pass in that game
GB_2020['receiver_player_id'] = GB_2020['receiver_player_id'].fillna(0)
GB_2020["receiver_player_id"].replace({"00-0031381": 1, "00-0034272": 2,
                                       "00-0033293": 3, "00-003375": 4,
                                       "00-0034521": 5, "00-0033948": 6,
                                       "00-0024243": 7, "00-0036265":8,
                                       "00-0034279": 9, "00-0033757": 10}, inplace=True)

#Dropoping receivers who did not have a play vs Buccs
values_rec = ["00-0032404", "00-0035671", "00-0035181", "00-0035480", "00-0036456", "00-0030525", "00-0036332", '00-0025580']
GB_2020 = GB_2020[GB_2020.receiver_player_id.isin(values_rec) == False]
print(GB_2020[['receiver_player_id', 'receiver_player_name']].value_counts())

# Select the desired features
features = GB_2020[["defense", "play_type_int", "rusher_player_id", "receiver_player_id",
                    "yardline_100", "down", "ydstogo"]]
print(features.head())

# Define Success
touchdown = GB_2020['successful_play']
print(touchdown)

# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    touchdown,
    test_size = 0.2,
    random_state = 1)

# Normalising the data, so it has a mean of 0 and standard deviation of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print(model.score(X_test_scaled, y_test))
# Analyze the coefficients
model.coef_     # a vector of the coefficients of each feature
print(list(zip(X_train.columns, model.coef_[0])))

# Creating Models to predict based on the features
pass_to_adams = [1,1,0,1,8,4,8]
pass_to_Valdes_Scantling = [1,1,0,2,8,4,8]
pass_to_lewis = [1,1,0,7,8,4,8]
pass_to_Jones = [1,1,0,3,8,4,8]
pass_to_Tonyan = [1,1,0,10,8,4,8]
pass_to_Lazard = [1,1,0,5,8,4,8]
pass_to_Williams = [1,1,0,6,8,4,8]
pass_to_St_Brown = [1,1,0,9,8,4,8]
pass_to_Dillon = [1,1,0,8,8,4,8]


run_by_jones = [1,0,1,0,8,4,8]
run_by_Williams = [1,0,2,0,8,4,8]
run_by_Dillon = [1,0,3,0,8,4,8]
run_by_Rodgers = [1,0,4,0,8,4,8]

# Combine arrays
sample_passes = np.array([pass_to_adams, pass_to_lewis, pass_to_Valdes_Scantling,pass_to_Jones,
                          pass_to_Tonyan, pass_to_Williams, pass_to_St_Brown, pass_to_Dillon, pass_to_Lazard])
sample_runs = np.array([run_by_jones, run_by_Williams, run_by_Dillon, run_by_Rodgers])

print(sample_runs)
# Scale the sample plays features
sample_passes_scaled = scaler.transform(sample_passes)
sample_runs_scaled = scaler.transform(sample_runs)
print(sample_passes_scaled)
print(sample_runs_scaled)

# Make scoring predictions
prediction_pass = model.predict(sample_passes_scaled)
prediction_run = model.predict(sample_runs_scaled)
probabilities_pass = model.predict_proba(sample_passes_scaled)
probabilities_run = model.predict_proba(sample_runs_scaled)

print('Passing predictions are : ' + str(prediction_pass))
print('Passing probabilities are : ' + str(probabilities_pass))
print('Running predictions are : ' + str(prediction_run))
print('Running probabilities are : ' + str(probabilities_run))

# Passing Plots
x = range(len(sample_passes))
y = probabilities_pass[:, 1]

x_2 = range(len(sample_runs))
y_2 = probabilities_run[:, 1]

plt.bar(x, y * 100, align="center")
plt.yticks(np.arange(0, 101, 25), ("0%", "25%", "50%", "75%", "100%"))
plt.xticks(np.arange(9), ("Davante Adams", "Valdes Scantling", "Mercedes Lewis", "Aaron Jones",
                          "Robert Tonyan", "Jamaal Williams", "Equanimeous St. Brown", "AJ Dillon", 'Allan Lazard'),
           rotation=45, ha='right', rotation_mode='anchor')

for xd,yd in zip(x,y):

    label = "{:.2f}".format(yd * 100)

    plt.annotate(label, # this is the text
                 (xd,yd), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title("Chance of Touchdown by Pass")

plt.tight_layout()
plt.show()
plt.cla()

# Running plots

plt.bar(x_2, y_2 * 100, align="center")
plt.yticks(np.arange(0, 101, 25), ("0%", "25%", "50%", "75%", "100%"))
plt.xticks(np.arange(4), ( "Aaron Jones", "Jamaal Williams","AJ Dillon", 'Aaron Rodgers'),
           rotation=45, ha='right', rotation_mode='anchor')
plt.title("Chance of Touchdown by Run")

for xr,yr in zip(x_2,y_2):

    label = "{:.2f}".format(yr * 100)

    plt.annotate(label, # this is the text
                 (xr,yr), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.tight_layout()
plt.show()