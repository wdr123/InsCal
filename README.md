# InsCal

Contains source code for paper submission "INSCAL: CALIBRATED MULTI-SOURCE FULLY TEST-TIME PROMPT TUNING FOR OBJECT DETECTION".


# Training

python tools/dist_train.sh path/to/config path/to/weights 4(8)


# Testing

python tools/dist_test.sh path/to/config path/to/weights 4(8)
