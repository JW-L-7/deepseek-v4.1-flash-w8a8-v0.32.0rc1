#!/usr/bin/env bash
set -euo pipefail
unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

NIC_NAME="<NETWORK_INTERFACE>"
LOCAL_IP="<DECODE_NODE_IP>"
MODEL_PATH="<YOUR_MODEL_PATH>"

export no_proxy="127.0.0.1,localhost,<PREFILL_NODE_IP>,<DECODE_NODE_IP>"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1800
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
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
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size $8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv41 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 400 \
    --max-num-seqs 32 \
    --async-scheduling \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.97 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config '{
        "kv_connector":"MultiConnector",
        "kv_role":"kv_consumer",
        "kv_port":"30100",
        "engine_id":"1",
        "kv_connector_extra_config":{
            "connectors":[
                {
                    "kv_connector":"MooncakeHybridConnector",
                    "kv_role":"kv_consumer",
                    "kv_port":"30100",
                    "kv_connector_extra_config":{
                        "prefill":{"dp_size":2,"tp_size":2},
                        "decode":{"dp_size":8,"tp_size":2}
                    }
                },
                {
                    "kv_connector":"AscendStoreConnector",
                    "kv_role":"kv_consumer",
                    "kv_connector_extra_config":{
                        "lookup_rpc_port":"0",
                        "backend":"mooncake",
                        "use_layerwise":false
                    }
                }
            ]
        }
    }' \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "enable_flashcomm1":false,
        "multistream_overlap_shared_expert":true,
        "recompute_scheduler_enable":true
    }'