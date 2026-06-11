# MiniSGL server arguments

Source file: `python/minisgl/server/args.py`

The parser is built in `parse_args(args, run_shell=False)`. The resulting dataclass is `ServerArgs`, which extends `SchedulerConfig`, which extends `EngineConfig`.

## Supported CLI arguments

| CLI argument                                  | Destination field                      | Type                     | Default       | Choices                                     | Meaning                                                                                                       |
| --------------------------------------------- | -------------------------------------- | ------------------------ | ------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--model-path`, `--model`                     | `model_path`                           | `str`                    | required      | any path or repo id                         | Model weight path. Can be a local folder or a Hugging Face repo ID.                                           |
| `--dtype`                                     | `dtype`                                | `str` then `torch.dtype` | `auto`        | `auto`, `float16`, `bfloat16`, `float32`    | Model weight and activation dtype. `auto` reads dtype from HF config, then maps strings to torch dtypes.      |
| `--tensor-parallel-size`, `--tp-size`         | `tensor_parallel_size`, then `tp_info` | `int`                    | `1`           | any int                                     | Tensor parallel world size. Converted to `DistributedInfo(0, tensor_parallel_size)`.                          |
| `--max-running-requests`                      | `max_running_req`                      | `int`                    | `256`         | any int                                     | Maximum number of concurrently running requests. （控制同时运行的请求槽位数量，主要决定page table的第一维大小 |
| `--max-seq-len-override`                      | `max_seq_len_override`                 | `int` or `None`          | `None`        | any int                                     | Overrides model maximum sequence length.（覆盖模型设置的序列最大输入长度）                                    |
| `--memory-ratio`                              | `memory_ratio`                         | `float`                  | `0.9`         | any float                                   | Fraction of GPU memory used when sizing KV cache.                                                             |
| `--dummy-weight`                              | `use_dummy_weight`                     | `bool`                   | `False`       | flag                                        | Use random dummy weights instead of loading real model weights.                                               |
| `--disable-pynccl`                            | `use_pynccl`                           | `bool`                   | `True`        | flag                                        | Disables PyNCCL tensor-parallel communication.                                                                |
| `--host`                                      | `server_host`                          | `str`                    | `127.0.0.1`   | any host                                    | API server host.                                                                                              |
| `--port`                                      | `server_port`                          | `int`                    | `1919`        | any port                                    | API server port. Also influences distributed init address through `server_port + 1`.                          |
| `--cuda-graph-max-bs`, `--graph`              | `cuda_graph_max_bs`                    | `int` or `None`          | `None`        | any int                                     | Maximum batch size for CUDA graph capture. `None` means auto tuning based on GPU memory.                      |
| `--num-tokenizer`, `--tokenizer-count`        | `num_tokenizer`                        | `int`                    | `0`           | any int                                     | Number of tokenizer processes. `0` means tokenizer is shared with detokenizer.                                |
| `--max-prefill-length`, `--max-extend-length` | `max_extend_tokens`                    | `int`                    | `8192`        | any int                                     | Chunked prefill maximum chunk size in tokens. Also used as scheduler max forward length.                      |
| `--num-pages`                                 | `num_page_override`                    | `int` or `None`          | `None`        | any int                                     | Overrides number of KV cache pages.                                                                           |
| `--page-size`                                 | `page_size`                            | `int`                    | `1`           | any int                                     | Page size for KV cache and system management.                                                                 |
| `--attention-backend`, `--attn`               | `attention_backend`                    | `str`                    | `auto`        | `auto`, `trtllm`, `fi`, `fa`, or `x,y` pair | Attention backend. A comma-separated pair means prefill backend and decode backend.                           |
| `--model-source`                              | `model_source`                         | `str`                    | `huggingface` | `huggingface`, `modelscope`                 | Model download source. Removed before `ServerArgs` is constructed.                                            |
| `--cache-type`                                | `cache_type`                           | `str`                    | `radix`       | `naive`, `radix`                            | Prefix/KV cache management strategy.                                                                          |
| `--moe-backend`                               | `moe_backend`                          | `str`                    | `auto`        | `auto`, `fused`                             | MoE backend. Used only when the model config is MoE.                                                          |
| `--shell-mode`                                | local `run_shell`                      | `bool`                   | `False`       | flag                                        | Runs server in shell mode. Removed before `ServerArgs` is constructed.                                        |

## Post-processing behavior

After `argparse` parses the CLI values, `parse_args()` applies these transformations:

- `--shell-mode` is popped into `run_shell`.
- When shell mode is enabled, it forces:
  - `cuda_graph_max_bs = 1`
  - `max_running_req = 1`
  - `silent_output = True`
- A `model_path` beginning with `~` is expanded with `os.path.expanduser`.
- If `--model-source modelscope` is used and `model_path` is not a local directory, `modelscope.snapshot_download()` is called.
- `model_source` is deleted before constructing `ServerArgs`.
- If `dtype == "auto"`, MiniSGL loads HF config with `cached_load_hf_config(model_path)` and reads its dtype.
- String dtype values are mapped to torch dtypes:
  - `float16` -> `torch.float16`
  - `bfloat16` -> `torch.bfloat16`
  - `float32` -> `torch.float32`
- `tensor_parallel_size` is converted to:
  - `tp_info = DistributedInfo(0, tensor_parallel_size)`
- `tensor_parallel_size` is deleted before constructing `ServerArgs`.

## Important inherited defaults

These defaults are not all directly declared in `server/args.py`, but many parser defaults reference them through `ServerArgs`.

From `EngineConfig`:

| Field                  | Default |
| ---------------------- | ------- |
| `max_running_req`      | `256`   |
| `attention_backend`    | `auto`  |
| `moe_backend`          | `auto`  |
| `cuda_graph_bs`        | `None`  |
| `cuda_graph_max_bs`    | `None`  |
| `page_size`            | `1`     |
| `memory_ratio`         | `0.9`   |
| `distributed_timeout`  | `60.0`  |
| `use_dummy_weight`     | `False` |
| `use_pynccl`           | `True`  |
| `max_seq_len_override` | `None`  |
| `num_page_override`    | `None`  |

From `SchedulerConfig`:

| Field               | Default |
| ------------------- | ------- |
| `max_extend_tokens` | `8192`  |
| `cache_type`        | `radix` |
| `offline_mode`      | `False` |

From `ServerArgs`:

| Field           | Default     |
| --------------- | ----------- |
| `server_host`   | `127.0.0.1` |
| `server_port`   | `1919`      |
| `num_tokenizer` | `0`         |
| `silent_output` | `False`     |

## Derived server addresses

`ServerArgs` also derives several IPC/TCP addresses:

| Property                       | Value                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| `zmq_backend_addr`             | `ipc:///tmp/minisgl_0.pid=<pid>`                                                     |
| `zmq_detokenizer_addr`         | `ipc:///tmp/minisgl_1.pid=<pid>`                                                     |
| `zmq_scheduler_broadcast_addr` | `ipc:///tmp/minisgl_2.pid=<pid>`                                                     |
| `zmq_frontend_addr`            | `ipc:///tmp/minisgl_3.pid=<pid>`                                                     |
| `zmq_tokenizer_addr`           | same as detokenizer when `num_tokenizer == 0`, else `ipc:///tmp/minisgl_4.pid=<pid>` |
| `distributed_addr`             | `tcp://127.0.0.1:<server_port + 1>`                                                  |

## Tokenizer sharing behavior

`num_tokenizer == 0` means `share_tokenizer` is true:

- tokenizer and detokenizer share the same ZMQ address
- `tokenizer_create_addr` is true
- backend/frontend do not create separate detokenizer/tokenizer links

`num_tokenizer > 0` means separate tokenizer workers are launched:

- detokenizer uses `zmq_detokenizer_addr`
- tokenizer workers use `zmq_tokenizer_addr`
- backend/frontend create separate tokenizer and detokenizer links
