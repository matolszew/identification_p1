import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from ar_model import ARmodel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Removing impulse interference from music recordings")
    parser.add_argument('filename', metavar='filename', type=str, help='path to wave file')
    parser.add_argument('r', metavar='r', type=int, default=4, help='order of AR model')
    parser.add_argument('buffer_size', metavar='buffer size', type=int, default=64, help='length of the window')
    parser.add_argument('buffer_hop', metavar='buffer hop', type=int, default=16, help='horizon of prediction')
    parser.add_argument('decay', metavar='decay', type=float, default=1.0, help='decay rate for exponential window')
    args = parser.parse_args()
    r = args.r
    n = args.buffer_size
    hop = args.buffer_hop

    fs, input = wavfile.read(args.filename)
    input = input / 2**15
    output = np.zeros_like(input)

    model = ARmodel(r, args.decay)

    for i in range(r, input.shape[0], hop):
        window_end = i+n
        if window_end > input.shape[0]:
            window_end = input.shape[0]
        hop_end = i+hop
        if hop_end > input.shape[0]:
            hop_end = input.shape[0]

        model.updateParams(input[i-r:window_end])
        output[i:hop_end] = model.estimateSignal(hop_end-i, input[i-r:i])

    # fig = plt.figure()
    # plt.plot(input[:1024], 'g')
    # plt.plot(output[:1024], 'r')
    # plt.savefig('plot.png')

    wavfile.write('out.wav', fs, output)
