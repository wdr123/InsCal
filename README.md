# CTOD

Contains source code for paper submission "Calibrated Multi-Source Fully Test-Time Prompt Tuning for Object Detection".


# Training

python tools/dist_train.sh path/to/config path/to/weights 4(8)


# Testing

python tools/dist_test.sh path/to/config path/to/weights 4(8)
