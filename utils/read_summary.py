import re
from pathlib import Path
import numpy as np

class SummaryEdf:
    def __init__(self, file_name, start_time, end_time, number_seizures, start_time_seizures, end_time_seizures):
        self.file_name = file_name
        self.start_time = start_time
        self.end_time = end_time
        self.number_seizures = number_seizures
        self.start_time_seizures = start_time_seizures
        self.end_time_seizures = end_time_seizures


# Press the green button in the gutter to run the script.
def get_seconds(string):
    match = re.search(r'\d+:\d+:\d+', string)
    if match:
        beg, end = match.span()
        v = string[beg:end]
        v = v.split(':')
        return 3600 * int(v[0]) + 60 * int(v[1]) + int(v[2])


def get_summary_edf(summary_path):
    
    list_summary = []
    # if not dataset_path.is_dir():
    #     dataset_path.mkdir(parents=True, exist_ok=False)

    file_name = None
    start_time = None
    end_time = None
    number_seizures = None
    start_time_seizures = []
    end_time_seizures = []

    # Variables to address the day reset problem
    t = 0
    d = 0
    end_time_raw = []

    with open(summary_path.__str__()) as f:
        # Start getting data
        for line in f.readlines():
            # variable to help addressing the day reset in start_time problem

            match = re.search(r'File Name:.+', line)
            if match:
                file_name = line[11:-1]  # avoid '\n'

            match = re.search(r'File Start Time:.+', line)
            if match:
                # https://strftime.org/
                # Addressing the day reset problem. If the end_time of previous interaction is higher than
                # the start_time of present interaction, add 1 day in seconds to start_time
                if t != 0 and get_seconds(line) < end_time_raw[t - 1]:
                    d += 1
                start_time = get_seconds(line) + 60 * 60 * 24 * d

            match = re.search(r'File End Time:.+', line)
            if match:
                # https://strftime.org/
                end_time_raw.append(get_seconds(line))
                end_time = get_seconds(line) + 60 * 60 * 24 * d

            match = re.search(r'Number of Seizures in File:.+', line)
            if match:
                match = re.search(r'[0-9]', line)
                number_seizures = int(match.group())

            if number_seizures != 0 and number_seizures != None:

                match = re.search(r'Seizure Start Time:.+', line)
                if match:
                    value = re.search(r'[0-9]+', line)
                    start_time_seizures.append(int(value.group()) + start_time)
                    
                match = re.search(r'Seizure End Time:.+', line)
                if match:
                    value = re.search(r'[0-9]+', line)
                    end_time_seizures.append(int(value.group()) + start_time)

                for i in range(number_seizures):
                    match = re.search(r'Seizure {} Start Time:.+'.format(i+1), line)
                    if match:
                        value = int(re.findall(r'[0-9]+', line)[-1])
                        #print(f'Entrei no start time da seizure {i+1} com valor: {value}')
                        start_time_seizures.append(value + start_time)
                            
                    match = re.search(r'Seizure {} End Time:.+'.format(i+1), line)
                    if match:
                        value = int(re.findall(r'[0-9]+', line)[-1])
                        #print(f'Entrei no end time da seizure {i+1} com valor: {value}')
                        end_time_seizures.append(value + start_time)
            
            
            # print(file_name, start_time, end_time, number_seizures)
            if file_name and start_time and end_time and (number_seizures is not None):
                if number_seizures == len(end_time_seizures):
                    list_summary.append(
                        SummaryEdf(file_name, start_time, end_time, number_seizures, start_time_seizures,
                                   end_time_seizures))

                    # Do lots of things!
                    # print('t value: ', t)
                    # print('file name: ', file_name)
                    # print('number of days: ', d)
                    # print('start time: ', start_time)
                    # print('end time: ', end_time)
                    # print('elaspsed time: ', end_time - start_time, '\n')
                    # print('number of seizures : ', number_seizures)
                    # print('start time of seizure : ', start_time_seizures)
                    # print('end time of seizure: ', end_time_seizures, '\n')

                    t += 1
                    file_name = None
                    start_time = None
                    end_time = None
                    number_seizures = None
                    start_time_seizures = []
                    end_time_seizures = []

    return list_summary
