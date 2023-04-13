import csv
from statistics import mean
import sys

if len(sys.argv) < 2:
    print("Please provide a filename as an argument.")
    sys.exit()

filename = sys.argv[1]

# Open the CSV file
with open(filename, 'r') as file:

    # Create a CSV reader object that returns a dictionary for each row
    csv_reader = csv.DictReader(file)
    array = []

    # Loop over each row in the CSV file
    for i, row in enumerate(csv_reader):
      if i == 0:
        # Print the value of the 'name' column for each row
        print(float(row['1_tend']) - float(row['1_tstart']))
    #   elif i < 8:
    #     array.append(float(row['1_tend']) - float(row['1_tstart']),)
    
    # print(mean(array), end=", ")
    # print(max(array))
