import csv

def readCSV(FILE):
    with open(FILE) as f:
        reader = csv.reader(f)
        list = [row for row in reader]
    return list


def outputCSV(rows, OUTPUTFILENAME):
    with open(OUTPUTFILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)