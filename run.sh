{
    set -x
    echo "$CONFIG_TYPE $N_W_DIM $N_DIM $LOG_DIR $0 $@"
    python -u x_evolve.py
} |& tee >(split -b 100M - run_api_${LOG_DIR}_part_ --additional-suffix=.log)