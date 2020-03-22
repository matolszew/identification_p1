import argparse
import numpy as np
from scipy.io import wavfile
from tqdm import trange

from ar_model import ARmodel

def correctSignal(signal, model, window_size, pred_size, step, treshold=3):
    """Correct signal using AR model

    Args:
        signal (np.array): signal to correct
        model (ARmodel): autoregresive model
        window_size (int): length of the window for updating AR model coefs
        pred_size (int): number of samples to generate from AR model
        step (int): step interval
        treshold (float): how many times error have to be bigger then standard
            deviation to classify sample as disturbed
    Returns:
        np.array: cerrected signal
    """
    out = np.copy(signal)

    for i in trange(0, input.shape[0]-window_size-pred_size, step):
        paramsEnd = i+window_size
        predEnd = paramsEnd+pred_size

        model.updateParams(out[i:paramsEnd])
        estimated = model.estimateSignal(pred_size, out[paramsEnd-model.r:paramsEnd])

        err = np.abs(out[paramsEnd:predEnd] - estimated)
        std = np.std(err)

        disturbed = np.abs(err) > std*treshold

        disturbanceLength = 0
        for j in range(pred_size):
            if disturbed[j]:
                disturbanceLength += 1
            elif disturbanceLength > 0:
                k = j + paramsEnd
                before = signal[k-disturbanceLength-1]
                after = signal[k]
                out[k-disturbanceLength:k] = np.linspace(before,after,disturbanceLength+2)[1:-1]
                disturbanceLength = 0

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Removing impulse interference from music recordings")
    parser.add_argument('filename', metavar='filename', type=str, help='path to wave file')
    parser.add_argument('-r', '--order', type=int, default=4, help='order of AR model')
    parser.add_argument('-o', '--out_file', type=str, default='out.wav', help='name of the output file')
    parser.add_argument('-u', '--param_window', type=int, default=256, help='length of the window for updating AR model coefs')
    parser.add_argument('-e', '--pred_widnow', type=int, default=8, help='number of samples to generate from AR model')
    parser.add_argument('-s', '--step', type=int, default=4, help='step interval')
    parser.add_argument('-d', '--decay', type=float, default=1.0, help='decay rate for exponential window')
    parser.add_argument('-m', '--max_std', type=float, default=3.0, help='how many times error have to be bigger then standard deviation to classify sample as disturbed')
    args = parser.parse_args()

    fs, input = wavfile.read(args.filename)
    input = input / 2**15

    model = ARmodel(args.order, args.decay)
    output = correctSignal(input, model, args.param_window, args.pred_widnow, args.step, args.max_std)

    wavfile.write(args.out_file, fs, output)
