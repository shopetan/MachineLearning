import csv

a = [[1,2],[3,4]]

f = open('a.csv','w')
writer = csv.writer(f)
writer.writerows(a)
f.close()
