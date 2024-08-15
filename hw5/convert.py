import csv
import sys

def convert(input_file, output_file):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    extracted_data = []
    for row in data:
        year = int(row['Winter'].split('-')[0].strip('"'))
        days = row['Days of Ice Cover'].strip('"')
        if days:
            days = int(days)
            extracted_data.append({'year': year, 'days': days})

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['year', 'days']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py inputFile outputFile")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert(input_file, output_file)
