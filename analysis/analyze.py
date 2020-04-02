import csv
import sys

with open(sys.argv[1]) as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  for row in reader:
    print("Post:\n")
    print(row[0] + "\n")
    print("Clarification Questions:\n")
    for i in range(1, 11):
        print("Q{}:  ".format(i) + row[i] + "\n")
