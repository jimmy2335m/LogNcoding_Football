import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_15 = pd.read_csv("./players_15.csv")
print(csv_15.head())

columns = csv_15.columns.tolist()
print(csv_15.columns.tolist())

names = csv_15['short_name'].head(10)
print(names)

overall_stat =[] 
highest_10 ={}
threshold = float("-inf")
for index,player in csv_15.iterrows():
    #print("{} :{}.format(player[short_name"], player["player_positions"]))
    player_stat = int(player['overall'])
    player_name = player["short_name"]
    player_potential = int(player["potential"])

    overall_stat.append(player_stat)
    #print(type(player_stat), type(tmp_min))
    if player_potential>threshold:
        print("Before threshold: {}, player_potential {}, highest_10 {}".format(threshold, player_potential, highest_10))
        highest_10[player_name] = player_potential
        print("After threshold: {}, player_potential {}, highest_10 {}".format(threshold, player_potential, highest_10))
        min_potential = min(highest_10.keys(), key=(lambda k: int(highest_10[k])))
        if len(highest_10)>10:
            del highest_10[min_potential]
            min_potential = min(highest_10.keys(), key=(lambda k: int(highest_10[k])))
            print("Before1 threshold: {}, player_potential {}, highest_10 {}, min_potential {}".format(threshold, player_potential, highest_10, min_potential))
            threshold = highest_10[min_potential]
            highest_10 = {key: value for key, value in sorted(highest_10.items(), key=lambda items: items[1], reverse=False)}
            print("After1 threshold: {}, player_potential {}, highest_10 {}, min_potential {}".format(threshold, player_potential, highest_10, min_potential))
print(highest_10)

mean_stat = np.round(np.mean(overall_stat), 3)
max_stat = np.max(overall_stat)
min_stat = np.min(overall_stat)
print("\nFrom {} players, mean stat: {}, max stat: {}, min stat: {}.".format(
    len(overall_stat), mean_stat, max_stat, min_stat))

fig = plt.figure()

plt.scatter(csv_15["overall"], csv_15["potential"])

#plt.show()

from sklearn.linear_model import LinearRegression


def linear_equation(filename):

    csv_filename = "./" + filename + ".csv"
    csv_file = pd.read_csv(csv_filename) #'./player_15.csv'

    X = csv_file['overall'].to_numpy()
    Y = csv_file["potential"].to_numpy()
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    print("X: ", X)
    print("Y: ", Y)

    reg = LinearRegression().fit(X, Y)
    score = reg.score(X, Y)
    print("score: ", score)

    coef = reg.coef_
    print('coef: ', coef)

    intercept = reg.intercept_
    print("intercept: ", intercept)

    print(type(coef))
    print(type(intercept))

    coef = coef.item()
    intercept = intercept.item()
    print(type(coef))
    print(type(intercept))

    print(coef)
    print(intercept)

    x = np.linspace(-10, 10, 100)
    y = coef * x + intercept 

    plt.cla()
    plt.plot(x, y, "-r", label="liner equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.grid()
    save_filename = "./linear_regression_" + filename + '.png'
    plt.savefig(save_filename, format="png")

    plt.show()

"""linear_equation('players_15')
linear_equation('players_16')
linear_equation('players_17')
linear_equation('players_18')
linear_equation('players_19')
linear_equation('players_20')"""

from sklearn.preprocessing import OneHotEncoder

categorical_features = ["nationality"]

enc = OneHotEncoder()

categorical_df = csv_15[categorical_features]

enc.fit(categorical_df)
print(enc.categories_)

ohe_df = enc.transform(categorical_df)
print(ohe_df)

