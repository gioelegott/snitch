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
    csv_reader = csv.reader(file)
    compute_time_array = []

    # Loop over each row in the CSV file
    for i, row in enumerate(csv_reader):
      if i == 1:
        # Print the value of the 'name' column for each row
        total_time = float(row[-1]) - float(row[1])
      elif i <= 9 and i > 1:
        compute_time_array.append(float(row[20]) - float(row[19]))
      elif i > 9:
        snitch_time = float(row[-4]) - float(row[7])
        data_tranfer_time = float(row[14]) - float(row[11]) + float(row[-5]) - float(row[-6])


    if len(sys.argv) > 2 and sys.argv[2] == '-x':
      print(total_time, end="       ")
      print(snitch_time, end="        ")
      print(max(compute_time_array), end="         ")
      print(data_tranfer_time)
    else:
      print(total_time)
