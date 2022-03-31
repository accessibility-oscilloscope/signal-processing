#!/usr/bin/env python3

import os
import argparse
import syslog
import numpy as np
# import matplotlib.pyplot as plt

INPUT_DATA_LENGTH = 4095
OUTPUT_DATA_LENGTH = 480

def normalize_signal(signal):
    sig_maxval = np.max(signal)
    sig_minval = np.min(signal)
    sig_norm = np.zeros(len(signal))
    for i in range(len(signal)):
        sig_norm[i] = (signal[i] - sig_minval) / (sig_maxval - sig_minval)
    return [sig_norm, sig_maxval, sig_minval]


def average(array):
    return array[0] * 0.45 + array[1] * 0.1 + array[2] * 0.45


def removed_anomalous_indexes(test_signal, anomalous_indexes):
    sorted_index = sorted(anomalous_indexes, reverse=True)
    array_len = len(sorted_index)
    for i in range(1, array_len - 1):
        test_signal[i] = np.uint8(average(test_signal[i - 1:i + 2]))
    return test_signal


def calculate_average_step(signal, threshold=3):
    """
    Determine the average step by doing a weighted average based on clustering of averages.
    array: our array
    threshold: the +/- offset for grouping clusters. Aplicable on all elements in the array.
    """

    # determine all the steps

    # for i in range(0, len(signal) - 1):
    steps = np.absolute(np.diff(signal))

    # determine the steps clusters
    clusters = []
    skip_indexes = []
    cluster_index = 0
    step_len = len(steps)
    for i in range(step_len):
        if i in skip_indexes:
            continue

        # determine the cluster band (based on threshold)
        cluster_lower = steps[i] - (steps[i] / 100) * threshold
        cluster_upper = steps[i] + (steps[i] / 100) * threshold

        # create the new cluster
        clusters.append([])
        clusters[cluster_index].append(steps[i])

        # try to match elements from the rest of the array
        for j in range(i + 1, step_len):

            if not (cluster_lower <= steps[j] <= cluster_upper):
                continue

            clusters[cluster_index].append(steps[j])
            skip_indexes.append(j)

        cluster_index += 1  # increment the cluster id

    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    biggest_cluster = clusters[0] if len(clusters) > 0 else None

    if biggest_cluster is None:
        return None

    return sum(biggest_cluster) / len(biggest_cluster)  # return our most common average


def detect_anomalous_values(array, regular_step, threshold=20):
    """
    Will scan every triad (3 points) in the array to detect anomalies.
    array: the array to iterate over.
    regular_step: the step around which we form the upper/lower band for filtering
    treshold: +/- variation between the steps of the first and median element and median and third element.
    """
    assert (len(array) >= 3)  # must have at least 3 elements

    anomalous_indexes = []

    step_lower = regular_step - (regular_step / 100) * threshold
    step_upper = regular_step + (regular_step / 100) * threshold

    # detection will be forward from i (hence 3 elements must be available for the d)
    absolute_diff = np.absolute(np.diff(array))

    array_len = len(array)
    for i in range(0, array_len - 2):
        a = array[i]
        b = array[i + 1]
        c = array[i + 2]

        first_step = absolute_diff[i]
        second_step = absolute_diff[i+1]

        first_belonging = step_lower <= first_step <= step_upper
        second_belonging = step_lower <= second_step <= step_upper

        # detect that both steps are alright
        if first_belonging and second_belonging:
            continue  # all is good here, nothing to do

        # detect if the first point in the triad is bad
        if not first_belonging and second_belonging:
            anomalous_indexes.append(i)

        # detect the last point in the triad is bad
        if first_belonging and not second_belonging:
            anomalous_indexes.append(i + 2)

        # detect the mid point in triad is bad (or everything is bad)
        if not first_belonging and not second_belonging:
            anomalous_indexes.append(i + 1)
            # we won't add here the others because they will be detected by
            # the rest of the triad scans

    return sorted(set(anomalous_indexes))  # return unique indexes

def downsample(array, output_size):
    # Example input: [1, 2, 3, 4], 2
    # Example output: [1.5, 3.5]

    div_amt = len(array) // output_size
    sub_size = len(array) // div_amt

    next_array = np.zeros(sub_size, dtype=np.uint8)
    for x in range(sub_size):
        next_array[x] = sum(array[x*div_amt: (x+1) * div_amt]) // div_amt
    return next_array


def scale_range(array):
    array += -(np.min(array))
    array *= 255 // np.max(array)
    return array


def process_data(signal):
    average_step = calculate_average_step(signal)
    anomalous_indexes = detect_anomalous_values(signal, average_step)
    test_signal = removed_anomalous_indexes(signal, anomalous_indexes)
    test_signal = downsample(test_signal, OUTPUT_DATA_LENGTH)
    test_signal = scale_range(test_signal)
    return test_signal


# def plot_data(new, old):
#     fig, ax = plt.subplots(2, 1, figsize=(20, 14))
#
#     ax[0].plot(range(len(old)), old, label='old')
#     ax[0].set_xlabel('Time')
#     ax[0].set_ylabel('Voltage')
#
#     ax[1].plot(range(len(new_data)), new, label='filtered')
#     ax[1].set_xlabel('Time')
#     ax[1].set_ylabel('Voltage')
#
#     fig.tight_layout()
#     fig.savefig('output.png', dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', metavar='i', dest='input_path', type=str, help='input')
    parser.add_argument('-o', metavar='o', dest='output_path', type=str, help='output')
    parser.add_argument('-v', dest='verbose', type=bool, default=False, help='verbose',
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    verbose = args.output_path
    if args.verbose:
        syslog.syslog("starting")
    input_fifo = os.open(input_path, os.O_RDONLY)
    output_fifo = os.open(output_path, os.O_WRONLY)

    if args.verbose:
        syslog.syslog("reading")

    binary_content_data = os.read(input_fifo, INPUT_DATA_LENGTH)
    if len(binary_content_data) != INPUT_DATA_LENGTH:
        syslog.syslog("Error: data is incorrect length: "+str(len(binary_content_data)))
        exit(1)

    if args.verbose:
        syslog.syslog("read "+str(len(binary_content_data))+" bytes")

    input_data = np.frombuffer(binary_content_data, dtype=np.uint8).copy()
    syslog.syslog(str(input_data))
    new_data = bytearray(process_data(input_data))

    # plot_data(new_data, input_data)
    if args.verbose:
        syslog.syslog("writing")
    output_length = os.write(output_fifo, new_data[0:OUTPUT_DATA_LENGTH])
    if len(new_data) != output_length:
        syslog.syslog("Entire buffer not written")
    if args.verbose:
        syslog.syslog("wrote "+str(len(new_data))+" bytes")

    os.close(input_fifo)
    os.close(output_fifo)

    exit(0)
