# MiniSGL scheduler notes

Source files:

- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/scheduler/prefill.py`
- `python/minisgl/scheduler/decode.py`
- `python/minisgl/scheduler/table.py`
- `python/minisgl/scheduler/cache.py`

## High-level behavior

MiniSGL's scheduler is a continuous batching scheduler.

It does not wait for a fixed batch of requests to fully finish before accepting the next group. Instead, every scheduler loop:

1. receives new messages if available
2. moves new user requests into the prefill pending queue
3. chooses the next batch to run
4. forwards that batch on the engine stream
5. processes the previous batch result
6. frees finished requests and keeps unfinished requests in the running decode set

The important consequence is that request membership in each GPU batch is dynamic. New requests can enter through prefill while older requests continue decode. Finished requests leave and free table/KV resources.

## Scheduler initialization

`Scheduler.__init__` creates these main components:

```python
self.engine = Engine(config)
self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
self.cache_manager = CacheManager(...)
self.decode_manager = DecodeManager(config.page_size)
self.prefill_manager = PrefillManager(...)
self.prefill_budget = config.max_extend_tokens
```

Roles:

- `Engine`: owns model forward, page table, KV cache pool, attention backend, sampler.
- `TableManager`: owns request slots and token pool.
- `CacheManager`: manages prefix cache and paged KV allocation.
- `PrefillManager`: queues new requests and schedules prefill batches.
- `DecodeManager`: tracks running decode requests and schedules decode batches.

## Main loop

The default loop is `overlap_loop` unless `ENV.DISABLE_OVERLAP_SCHEDULING` is enabled.

Pseudo-flow:

```text
while True:
  receive frontend/backend messages
  add new UserMsg into prefill pending list
  schedule next batch:
    prefill batch if available and resources allow
    otherwise decode batch
  run forward on engine stream
  process previous batch results
```

The overlap path intentionally overlaps CPU-side scheduling/result processing with GPU-side forward work:

```python
ongoing_data = (forward_input, self._forward(forward_input))
self._process_last_data(last_data)
```

## Scheduling policy

The current policy is prefill-first:

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

Meaning:

- If prefill can produce a batch, scheduler runs prefill.
- If no prefill batch can run, scheduler runs decode for current running requests.
- A TODO in code says other policies such as decode-first are not implemented yet.

This means new prompts can temporarily take priority over decode, subject to resource checks.

## Prefill path

New `UserMsg` requests are appended into `PrefillManager.pending_list`.

When scheduling prefill:

1. `PrefillAdder` is created with:
   - `token_budget = max_extend_tokens`
   - `reserved_size = decode_manager.inflight_tokens`
2. It scans `pending_list` in order.
3. For each pending request, it checks:
   - whether a table slot is available
   - whether KV cache has enough space
   - whether token budget remains
4. If accepted, it allocates a table slot and creates a `Req`.
5. If prompt is longer than current prefill token budget, it creates a `ChunkedReq`.

The key budget is:

```python
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
```

So `--max-prefill-length` / `--max-extend-length` controls how many prompt tokens one prefill scheduling round may admit.

## Chunked prefill

Long prompts may be split across multiple scheduler rounds.

When a request cannot fit fully under the current prefill token budget:

- it becomes a `ChunkedReq`
- it is not added to decode yet
- it is put back into the pending list with progress saved
- later prefill rounds continue from its cached length

`ChunkedReq.can_decode` returns `False`, so chunked prefill pieces avoid entering the decode manager.

## Decode path

`DecodeManager` maintains:

```python
running_reqs: Set[Req]
```

After a prefill or decode forward, `_forward()` calls:

```python
self.decode_manager.filter_reqs(forward_input.batch.reqs)
```

That unions newly forwarded requests with existing running requests and keeps only requests where `req.can_decode` is true.

When decode is scheduled:

```python
return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")
```

So each decode batch contains all currently runnable decode requests, sorted by uid. Each decode forward normally generates one token per request.

## Request completion and resource release

After a batch result is available, `_process_last_data()` checks each non-chunked request:

- append generated token to host-side request state
- mark finished if `req.can_decode` is false
- also mark finished if EOS is generated and `ignore_eos` is false
- send a detokenization message

When finished:

```python
self.decode_manager.remove_req(req)
self._free_req_resources(req)
```

Resource release:

```python
self.table_manager.free(req.table_idx)
self.cache_manager.cache_req(req, finished=True)
```

So completion returns the request slot and releases/caches KV resources.

## Request count control

Continuous batching does not mean unlimited requests.

`--max-running-requests` controls the number of request table slots:

```python
self._free_slots = list(range(max_running_reqs))
```

Before admitting a prefill request:

```python
if self.table_manager.available_size == 0:
    return None
```

Engine page table is sized as:

```python
(config.max_running_req + 1, aligned_max_seq_len)
```

The extra one row is for the dummy request used by graph capture.

Actual running capacity is also limited by KV cache:

```python
estimated_len = extend_len + req.output_len
if estimated_len + reserved_size > self.cache_manager.available_size:
    return None
```

So the effective number of simultaneously running requests is bounded by:

```text
min(max_running_requests, KV cache capacity, request length mix)
```

## Sequence length control

`--max-seq-len-override` controls the configured maximum sequence length.

The final engine max sequence length is:

```python
self.max_seq_len = min(config.max_seq_len, num_tokens)
```

So it is also capped by the number of KV-cache-backed tokens.

When a request arrives:

```python
input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
max_output_len = max_seq_len - input_len
```

If `max_output_len <= 0`, the request is dropped. If requested output tokens exceed the remaining sequence budget, `max_tokens` is truncated.

## Important parameters

| Parameter | Effect |
| --- | --- |
| `--max-running-requests` | Upper bound for active request slots. Also sizes page table and token pool first dimension. |
| `--max-seq-len-override` | Overrides model max sequence length, then final value is capped by KV cache capacity. |
| `--max-prefill-length`, `--max-extend-length` | Prefill token budget per scheduling round. Controls chunked prefill behavior. |
| `--num-pages` | Overrides KV cache page count. Can increase or decrease effective running capacity. |
| `--page-size` | KV cache page size. Also affects reserved decode tokens. |
| `--cache-type` | Prefix cache implementation, currently `radix` or `naive`. |

## Practical mental model

MiniSGL scheduler has two queues/sets:

```text
pending prefill list:
  new requests and unfinished prompt chunks

running decode set:
  requests that finished prefill and still need output tokens
```

Every loop picks:

```text
prefill if possible, otherwise decode
```

This is continuous batching because the decode set changes over time and new prefill requests can be admitted between decode rounds. It is not fixed static batching.
