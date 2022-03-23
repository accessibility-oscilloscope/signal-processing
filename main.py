import os
import argparse
import numpy as np
import syslog
# import matplotlib.pyplot as plt

DATA_LENGTH = 480


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
    for i in range(1, len(sorted_index) - 1):
        test_signal[i] = int(average(test_signal[i - 1:i + 2]))
    return test_signal


def calculate_average_step(signal, threshold=3):
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

        # detect the last point in the triad is bad
        if first_belonging and not second_belonging:
            anomalous_indexes.append(i + 2)

        # detect the mid point in triad is bad (or everything is bad)
        if not first_belonging and not second_belonging:
            anomalous_indexes.append(i + 1)
            # we won't add here the others because they will be detected by
            # the rest of the triad scans

    return sorted(set(anomalous_indexes))  # return unique indexes


def process_data(input_data):
    test_signal = [x for x in input_data]

    average_step = calculate_average_step(test_signal)
    anomalous_indexes = detect_anomalous_values(test_signal, average_step)
    test_signal = removed_anomalous_indexes(test_signal, anomalous_indexes)

    # filter_to_single_period(test_signal)
    return test_signal


def filter_to_single_period(test_signal):
    dt = 1
    signal = np.array(test_signal) / 255
    # dt = 0.001
    # t = np.arange(0, 1, dt)
    # signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # composite signal
    t = range(len(signal))
    signal_clean = signal
    minsignal, maxsignal = signal.min(), signal.max()
    ## Compute Fourier Transform
    n = len(t)
    fhat = np.fft.fft(signal, n)  # computes the fft
    psd = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)  # frequency array
    idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)  # first half index
    ## Filter out noise
    threshold = 100
    psd_idxs = psd > threshold  # array of 0 and 1
    psd_clean = psd * psd_idxs  # zero out all the unnecessary powers
    fhat_clean = psd_idxs * fhat  # used to retrieve the signal
    signal_filtered = np.fft.ifft(fhat_clean)  # inverse fourier transform
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, signal, color='b', lw=0.5, label='Noisy Signal')
    ax[0].plot(t, signal_clean, color='r', lw=1, label='Clean Signal')
    ax[0].set_ylim([minsignal, maxsignal])
    ax[0].set_xlabel('t axis')
    ax[0].set_ylabel('Vals')
    ax[0].legend()
    ax[1].plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=0.5, label='PSD noisy')
    ax[1].set_xlabel('Frequencies in Hz')
    ax[1].set_ylabel('Amplitude')
    ax[0].set_ylim([minsignal, maxsignal])
    ax[1].legend()
    ax[2].plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
    ax[2].set_xlabel('Frequencies in Hz')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_ylim([min(np.abs(psd_clean[idxs_half])), max(np.abs(psd_clean[idxs_half]))])
    ax[2].legend()
    ax[3].plot(t, signal_filtered, color='r', lw=1, label='Clean Signal Retrieved')
    ax[3].set_ylim([minsignal, maxsignal])
    ax[3].set_xlabel('t axis')
    ax[3].set_ylabel('Vals')
    ax[3].legend()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('signal-analysis.png', bbox_inches='tight', dpi=300)


def plot_data(new, old):
    fig = plt.figure(figsize=(10, 7), dpi=300)
    plt.plot(range(len(old)), old, label='yolo method')
    plt.plot(range(len(new_data)), new, label='yolo1 method')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('output.png')


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
        print("starting")
    input_fifo = os.open(input_path, os.O_RDONLY)
    output_fifo = os.open(output_path, os.O_WRONLY)

    if args.verbose:
        print("reading")

    binary_content_data = os.read(input_fifo, DATA_LENGTH)
    if len(binary_content_data) != DATA_LENGTH:
        exit(1)

    if args.verbose:
        print("read "+str(len(binary_content_data))+" bytes")
    input_data = list(binary_content_data)
    new_data = bytearray(process_data(input_data))

    if args.verbose:
        print("writing")
    os.write(output_fifo, new_data)
    if args.verbose:
        print("wrote "+str(len(new_data))+" bytes")

    os.close(input_fifo)
    os.close(output_fifo)

    exit(0)
