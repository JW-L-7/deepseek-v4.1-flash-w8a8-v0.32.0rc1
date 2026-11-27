#!/usr/bin/env bash
set -euo pipefail
unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<PREFILL_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"

export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_IP>,<DECODE_NODE_IP>"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export ASCEND_RT_VISIBLE_DEVICES=$1

export MOONCAKE_CONFIG_PATH="/model/mooncake.json"
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${{70 * 1024 * 1024 * 1024}}
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=${{70 * 1024 * 1024 * 1024}}
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY="lru"
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824

echo 3 > /proc/sys/vm/drop_caches
echo 300000 > /proc/sys/vm/nr_hugepages

export MC_STORE_USE_HUGEPAGE=1
export MC_STORE_HUGEPAGE_SIZE=2MB

vllm serve "$MODEL_PATH" \
    --device-ids "$1" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-size-local 1 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size $8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
    --trust-remote-code \
    --block-size 128 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.97 \
    --quantization ascend \
    --enforce-eager \
    --swa-bounded-replay \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "enable_cpu_binding":true,
        "enable_fused_mc2":1,
        "enable_flashcomm1":false,
        "enable_shared_expert_dp":true
    }' \
    --headless \
    --data-parallel-start-rank 1 \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_producer",
        "kv_port":"30000",
        "engine_id":"0",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_producer",
                    "kv_port":"30000",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":2,"tp_size":2},
                        "decode":{"dp_size":8,"tp_size":2}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_producer",
                    "kv_connector_extra_config":{
                        "lookup_rpc_port":"0",
                        "backend":"mooncake",
                        "use_layerwise":false,
                        "save_decode_cache":true
                    }
                }
            ]
        }
    }'