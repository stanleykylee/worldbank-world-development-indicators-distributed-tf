import csv
import json

# configuration
DATA_FILE = 'WDI_Data.csv'
INDICATORS_FILE = 'indicators.config'
OUTPUT_FILE = 'time-series.csv'

def make_country_dict():
    country = {} 
    for i in range(0,57):
        country[i] = {}
    return country

# extract selected indicators and write time series entries of them to csv
def flush(dict):
    out_str = ''
    for entry in dict:
        if len(dict[entry]) < len(selected_indicators):
            continue
        out_str = ''
        for key in dict[entry]:
            out_str += dict[entry][key] + ','
        out_str = out_str[:-1] + '\n'
        with open(OUTPUT_FILE, 'a') as f:
            f.write(out_str)
            f.flush()
    return

# create list of indicators selected from dataset
with open(INDICATORS_FILE) as f:
    selected_indicators = f.readlines()
selected_indicators = [elem.strip() for elem in selected_indicators] 

with open(DATA_FILE, 'rb') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    idx = 0
    for row in csv_reader:
        if (idx == 0):
            idx += 1
            continue;
        if (idx == 1):
            country_dict = make_country_dict()
            country = row[0]
        if (row[0] != country):
            country = row[0]
            flush(country_dict)
            country_dict = make_country_dict()
        row_idx = 0
        row_name = row[3]
        if row_name in selected_indicators:
            for item in row:
                if (row_idx > 3 and item != ''):
                    country_dict[row_idx - 4][row_name] = item
                row_idx += 1
        idx += 1
