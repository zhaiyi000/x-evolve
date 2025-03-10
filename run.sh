{
    set -x
    python -u funsearch_bin_packing_llm_api.py
} |& tee run_api_$LOG_DIR.log