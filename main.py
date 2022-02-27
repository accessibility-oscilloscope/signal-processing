import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_signal = [4, 5, 6, 7, 6, 5]


def normalize_signal(signal):
    sig_maxval = np.max(signal)
    sig_minval = np.min(signal)
    sig_norm = np.zeros(len(signal))
    for i in range(len(signal)):
        sig_norm[i] = (signal[i] - sig_minval) / (sig_maxval - sig_minval)
    return [sig_norm, sig_maxval, sig_minval]

def removed_anomalous_indexes(test_signal, anomalous_indexes):
    print(test_signal)
    print(anomalous_indexes)
    for x in sorted(anomalous_indexes, reverse=True):
        test_signal[x] = test_signal[x - 1]
    return test_signal

def calculate_average_step(signal, threshold=15):
    """
    Determine the average step by doing a weighted average based on clustering of averages.
    array: our array
    threshold: the +/- offset for grouping clusters. Aplicable on all elements in the array.
    """

    # determine all the steps
    steps = []
    for i in range(0, len(signal) - 1):
        steps.append(abs(signal[i] - signal[i + 1]))

    # determine the steps clusters
    clusters = []
    skip_indexes = []
    cluster_index = 0

    for i in range(len(steps)):
        if i in skip_indexes:
            continue

        # determine the cluster band (based on threshold)
        cluster_lower = steps[i] - (steps[i] / 100) * threshold
        cluster_upper = steps[i] + (steps[i] / 100) * threshold

        # create the new cluster
        clusters.append([])
        clusters[cluster_index].append(steps[i])

        # try to match elements from the rest of the array
        for j in range(i + 1, len(steps)):

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


def detect_anomalous_values(array, regular_step, threshold=25):
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
    for i in range(0, len(array) - 2):
        a = array[i]
        b = array[i + 1]
        c = array[i + 2]

        first_step = abs(a - b)
        second_step = abs(b - c)

        first_belonging = step_lower <= first_step <= step_upper
        second_belonging = step_lower <= second_step <= step_upper

        # detect that both steps are alright
        if first_belonging and second_belonging:
            continue  # all is good here, nothing to do

        # detect if the first point in the triad is bad
        if not first_belonging and second_belonging:
            anomalous_indexes.append(i)
            print("first point bad")

        # detect the last point in the triad is bad
        if first_belonging and not second_belonging:
            anomalous_indexes.append(i + 2)
            print("last point bad")

        # detect the mid point in triad is bad (or everything is bad)
        if not first_belonging and not second_belonging:
            anomalous_indexes.append(i + 1)
            print("mid point")
            # we won't add here the others because they will be detected by
            # the rest of the triad scans

    return sorted(set(anomalous_indexes))  # return unique indexes

try:
    f = open(r"C:\Users\Alex\Desktop\Accessible Oscilloscope\Signal Processing\test.signal", "rb")
    while True:
        binary_content = f.read(-1)
        if not binary_content:
            break
        binary_content_data = binary_content
except IOError:
    print("error")
test_signal = [x for x in binary_content_data]
# plt.plot(range(len(test_signal)), test_signal)

average_step = calculate_average_step(test_signal)
anomalous_indexes = detect_anomalous_values(test_signal, average_step)

print(anomalous_indexes)
test_signal = removed_anomalous_indexes(test_signal, anomalous_indexes)
[sig_norm, sig_maxval, sig_minval] = normalize_signal(test_signal)
max_index = test_signal.index(sig_maxval)
min_index = test_signal.index(sig_minval)

half_period = abs(max_index - min_index)
if max_index > min_index:
    final_array = test_signal[min_index - int(half_period / 2): max_index + int(half_period / 2)]
else:
    final_array = test_signal[max_index - int(half_period / 2): min_index + int(half_period / 2)]




plt.plot(range(len(final_array)), final_array)
plt.show()