DECODE_NODE_IP="<DECODE_NODE_IP>"

python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 2 \
    --pcp-size 1 \
    --dp-size-local 8 \
    --dp-rank-start 0 \   #当前机器DP rank的起始值。若D节点一共有8个DP rank，且当前机器是第1台机器，则dp-rank-start=0；若当前机器是第2台机器，则dp-rank-start=4
    --dp-address "$DECODE_NODE_IP" \
    --dp-rpc-port 12321 \
    --vllm-start-port 1025   #DP的起始端口，不是推理服务的预测端口