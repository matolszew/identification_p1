# identification_p1
Removing impulse interference from music recordings with on the autoregressive model. Model parameters are calculated using weighted least square algorithm. Samples are classified as disturbed when error of estimation with the AR model is larger than given standard deviation. Corrected values in disturbed samples are calculated using linear regression between samples before and after interference.

## Usage eaxmple
```
python main.py recording.wav
```

### Parameters description
```
python main.py -h
usage: main.py [-h] [-r ORDER] [-o OUT_FILE] [-u PARAM_WINDOW] [-e PRED_WIDNOW] [-s STEP] [-d DECAY] [-m MAX_STD] filename

Removing impulse interference from music recordings

positional arguments:
  filename              path to wave file

optional arguments:
  -h, --help            show this help message and exit
  -r ORDER, --order ORDER
                        order of AR model
  -o OUT_FILE, --out_file OUT_FILE

                        name of the output file
  -u PARAM_WINDOW, --param_window PARAM_WINDOW
                        length of the window for updating AR model coefs
  -e PRED_WIDNOW, --pred_widnow PRED_WIDNOW
                        number of samples to generate from AR model
  -s STEP, --step STEP  step interval
  -d DECAY, --decay DECAY
                        decay rate for exponential window
  -m MAX_STD, --max_std MAX_STD
                        how many times error have to be bigger then standard deviation to classify sample as disturbed
```

Project no 1 on Processes Identification on Control Engineering and Robotics master degree course on Gdańsk University of Technology.
