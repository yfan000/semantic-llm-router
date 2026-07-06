"""ncompare_all.py — Side-by-side comparison of all routing approaches.

Produces a single unified table ranking every system by accuracy, latency,
and cost — both overall and broken down by (domain, complexity).

Usage:
    python tests/compare_all.py \\
        --system "TTCA+retry:results/router_ttca.csv" \\
        --system "Cascade:results/baseline_cascade.csv" \\
        --system "vLLM-SR:results/baseline_vllm_sr.csv" \\
        --system "Round-Robin:results/rr_baseline.csv" \\
        --eval-matrix results/eval_matrix.csv \\
        --output results/compare_all.csv

Each --system argument is NAME:PATH.  Order determines the column order in the
domain breakdown table.  Systems are ranked by overall accuracy in the summary.
"""