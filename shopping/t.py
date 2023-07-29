import csv
import sys
import pandas as pd

month_to_int = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")
    
    evidence, labels = load_data(sys.argv[1])
    

def load_data(filename):


    raw_data = pd.read_csv(filename, header=0).values.tolist()

    #duplicate to evidence from raw_data with last item from every list removed
    evidence = [row[:-1] for row in raw_data]
    
    #duplicate to label from raw_data and change from boolean to int
    label = [int(row[-1]) for row in raw_data]

    #change evidence items to floats and integers
    for row in evidence:
        row[1] = float(row[1])
        row[3] = float(row[3])
        row[5] = float(row[5])
        row[6] = float(row[6])
        row[7] = float(row[7])
        row[8] = float(row[8])
        row[9] = float(row[9])
        
        row[10] = month_to_int[row[10]]
        if row[-2] == "Returning_Visitor":
            row[-2] = 1
        else:
            row[-2] = 0
        
        if row[-1] is True:
            row[-1] = 1
        else:
            row[-1] = 0

    return (evidence, label)



    


if __name__ == "__main__":
    main()