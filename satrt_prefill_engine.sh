PREFILL_NODE_IP="<PREFILL_NODE_IP>"

python launch_online_dp.py \
    --dp-size 2 \
    --tp-size 2 \
    --pcp-size 4 \
    --dp-size-local 1 \
    --dp-rank-start 0 \
    --dp-address "$PREFILL_NODE_IP" \
    --dp-rpc-port 12321 \
    --vllm-start-port 1025