from sklearn.ensemble import RandomForestClassifier

trainingdata = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
traininglabel = [1, 1, -1, -1]
testdata = [[3, 3], [-3, -3]]

model = RandomForestClassifier()
model.fit(trainingdata, traininglabel)
output = model.predict(testdata)

for label in output: print label
