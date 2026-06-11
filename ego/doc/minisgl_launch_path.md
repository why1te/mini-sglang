# MiniSGL launch path for `python -m minisgl --model`

Command example:

```bash
python -m minisgl --model "/data/you/models/..."
```

## Summary

`--model` is an alias of `--model-path`. After argument parsing, the value is stored as `ServerArgs.model_path`.

That path is used in two major places:

- Model side: scheduler creates an `Engine`, which reads Hugging Face config, creates the runtime model class, and loads `.safetensors` weights.
- Tokenizer side: tokenizer and detokenizer worker processes call `AutoTokenizer.from_pretrained(model_path)`.

## Main call chain

1. Python module entrypoint

   File: `python/minisgl/__main__.py`

   ```python
   from .server import launch_server
   launch_server()
   ```

2. Server launch

   File: `python/minisgl/server/launch.py`

   `launch_server()` calls:

   ```python
   server_args, run_shell = parse_args(sys.argv[1:], run_shell)
   run_api_server(server_args, start_subprocess, run_shell=run_shell)
   ```

3. Argument parsing

   File: `python/minisgl/server/args.py`

   ```python
   parser.add_argument("--model-path", "--model", ...)
   ```

   The parsed field is `model_path`.

4. Worker process startup

   File: `python/minisgl/server/launch.py`

   `start_subprocess()` starts:

   - one scheduler process per tensor-parallel rank
   - one detokenizer process
   - optional tokenizer processes controlled by `--num-tokenizer`

   Scheduler target:

   ```python
   _run_scheduler(new_args, ack_queue)
   ```

   Tokenizer path:

   ```python
   "tokenizer_path": server_args.model_path
   ```

5. Scheduler initialization

   File: `python/minisgl/scheduler/scheduler.py`

   ```python
   self.engine = Engine(config)
   self.tokenizer = load_tokenizer(config.model_path)
   ```

6. Engine initialization

   File: `python/minisgl/engine/engine.py`

   ```python
   with torch.device("meta"), torch_dtype(config.dtype):
       self.model = create_model(config.model_config)
   self.model.load_state_dict(self._load_weight_state_dict(config))
   ```

7. Hugging Face config to MiniSGL model config

   File: `python/minisgl/engine/config.py`

   ```python
   hf_config = cached_load_hf_config(self.model_path)
   model_config = ModelConfig.from_hf(hf_config)
   ```

8. Model class selection

   File: `python/minisgl/models/register.py`

   The runtime model class is selected from `model_config.architectures[0]`.

   Supported architecture mappings:

   - `LlamaForCausalLM` -> `python/minisgl/models/llama.py`
   - `Qwen2ForCausalLM` -> `python/minisgl/models/qwen2.py`
   - `Qwen3ForCausalLM` -> `python/minisgl/models/qwen3.py`
   - `Qwen3MoeForCausalLM` -> `python/minisgl/models/qwen3_moe.py`
   - `MistralForCausalLM` -> `python/minisgl/models/mistral.py`
   - `Mistral3ForConditionalGeneration` -> `python/minisgl/models/mistral.py`

9. Weight loading

   File: `python/minisgl/models/weight.py`

   ```python
   model_folder = download_hf_weight(model_path)
   files = glob.glob(f"{model_folder}/*.safetensors")
   ```

   File: `python/minisgl/utils/hf.py`

   ```python
   if os.path.isdir(model_path):
       return model_path
   ```

   For a local path such as `/data/you/models/...`, MiniSGL directly uses that local directory and loads `.safetensors` files from it.

10. Tokenizer loading

    File: `python/minisgl/utils/hf.py`

    ```python
    AutoTokenizer.from_pretrained(model_path)
    ```

## Condensed flow

```text
python -m minisgl
 -> python/minisgl/__main__.py
 -> server.launch.launch_server()
 -> server.args.parse_args()
    --model => ServerArgs.model_path
 -> server.api_server.run_api_server(...)
 -> launch.py:start_subprocess()
    -> scheduler process: Scheduler(args)
       -> Engine(config)
          -> AutoConfig.from_pretrained(model_path)
          -> ModelConfig.from_hf(...)
          -> create_model(...)
          -> load_weight(model_path, device)
    -> tokenizer process: tokenize_worker(tokenizer_path=model_path)
       -> AutoTokenizer.from_pretrained(model_path)
```
