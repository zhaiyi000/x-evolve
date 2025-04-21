{
    set -x
    echo "$CONFIG_TYPE $N_DIM $LOG_DIR $0 $@"
    python -u funsearch_bin_packing_llm_api.py
} |& tee >(split -b 100M - run_api_${LOG_DIR}_part_ --additional-suffix=.log)