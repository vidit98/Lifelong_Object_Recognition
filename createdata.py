import os
import csv

for i in range(1, 13):
	task = "task" + str(i)
	dire = "test/" + task
	l = [x[0] for x in os.walk("test/" + task)]
	l.sort()
	print(l)
	with open("test/" + task + "/cabels.csv", "w") as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(["file name", "label"])
		for i in range(1, len(l)):
			print(l[i])
			files = [x[2] for x in os.walk(l[i])]
			# print(files)
			for f in files[0]:
				writer.writerow([l[i] + "/"+ f, str(i-1)])

	csvFile.close()
