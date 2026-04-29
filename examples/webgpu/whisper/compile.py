import sys, pathlib, argparse, json, base64
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from collections import OrderedDict
from tinygrad import Tensor, Variable, dtypes
from tinygrad.helpers import DEV, Context
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.engine.jit import TinyJit
from tinygrad.uop.ops import Ops
from extra.export_model import export_model, compile_net, export_model_webgpu

from examples.whisper import init_whisper, LANGUAGES
from examples.webgpu.whisper.melspec import MelSpec, SAMPLES_PER_SEGMENT, N_MELS, FRAMES_PER_SEGMENT

def save(out, name, prg, state):
  (out/f"net_{name}.js").write_text(prg)
  safe_save(state, (out/f"net_{name}.safetensors").as_posix())
  mb = sum(t.nbytes() for t in state.values()) / (1024*1024)
  print(f"[{name}] js={len(prg)//1024} KiB safetensors={mb:.1f} MiB")

def dump_vocab(enc, path):
  out = {}
  for i in range(enc.n_vocab):
    try: out[str(i)] = {"t": enc.decode_single_token_bytes(i).decode("utf-8")}
    except UnicodeDecodeError: out[str(i)] = {"b": base64.b64encode(enc.decode_single_token_bytes(i)).decode("ascii")}
    except KeyError: continue
  path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')))

# decoder: symbolic pos, seqlen=1. export_model would run capture twice with identical args so
# pos gets baked as a const; drive the two runs ourselves with distinct bindings.
def export_decoder(model, out):
  D = model.decoder.token_embedding.weight.shape[1]
  encoded = Tensor.randn(1, FRAMES_PER_SEGMENT//2, D, dtype=dtypes.float32).realize()
  tokens = Tensor.randint(1, 1, low=0, high=1000, dtype=dtypes.int32).realize()
  pos = Variable("self_attn_cache_len", 0, model.decoder.max_self_attn_cache_len-1)
  @TinyJit
  def run(t, e, p): return [model.decoder.forward(t, p, e).realize()]
  with Context(JIT=2, CPU_COUNT=1):
    run(tokens, encoded, pos.bind(1))
    out_bufs = run(tokens, encoded, pos.bind(2))
  fns, stmts, bufs, _ = compile_net(run.captured.linear, [o.uop.base.realized for o in out_bufs])
  # strip KV caches: capture populated them with sample data and WebGPU zero-inits bufs anyway
  state = {f"decoder.{k}":v for k,v in get_state_dict(model.decoder).items() if ".cache_" not in k and v.uop.base.realized is not None}
  names = {(id(b), b.offset, b.size, b.dtype): k for k,v in state.items() if (b:=v.uop.base.realized)}
  # symbolic-var replay: lift DEFINE_VARs out of kernel args and ADD-const global_size expressions
  sv = OrderedDict()
  for i, (_, a, gs, _) in enumerate(stmts):
    for j, v in enumerate(a):
      if getattr(v, "op", None) is Ops.DEFINE_VAR and isinstance(getattr(v, "arg", None), tuple):
        if v not in sv: sv[v] = v.arg[0]; bufs[v.arg[0]] = (v.dtype.itemsize, v.dtype, v.arg[0])
        stmts[i][1][j] = sv[v]
    for j, d in enumerate(gs or ()):
      if getattr(d, "op", None) is Ops.ADD and len(d.src) == 2 and {d.src[0].op, d.src[1].op} == {Ops.DEFINE_VAR, Ops.CONST}:
        nm, val = d.src if d.src[1].op is Ops.CONST else tuple(reversed(d.src))
        gs[j] = f"_{nm.arg[0]}[0] + {val.arg}"
  prg = export_model_webgpu(fns, stmts, bufs, names, ["input0","input1"], [f"output{i}" for i in range(len(out_bufs))], "decoder", sv)
  save(out, "decoder", prg, state)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="tiny.en", choices=["tiny.en","tiny","base.en","base","small.en","small"])
  args = parser.parse_args()
  DEV.value = "WEBGPU"
  out = pathlib.Path(__file__).parent

  model, enc = init_whisper(args.model, batch_size=1)
  # OpenAI checkpoint is f16; mixed f16/f32 kernels diverge between Chrome-Dawn and Python-Dawn
  # (root cause: ~0.55% of trained weights are denormal f16, and runtimes disagree on whether
  # to flush them per WGSL spec; compounded by FMA/reduction-order spec freedom)
  for t in get_state_dict(model).values():
    if t.dtype == dtypes.float16: t.replace(t.cast(dtypes.float32).realize())

  prg, _, _, state = export_model(MelSpec(), "webgpu", Tensor.randn(1, SAMPLES_PER_SEGMENT, dtype=dtypes.float32).realize(), model_name="melSpec")
  save(out, "melSpec", prg, state)

  prg, _, _, state = export_model(model.encoder, "webgpu", Tensor.randn(1, N_MELS, FRAMES_PER_SEGMENT, dtype=dtypes.float32).realize(), model_name="encoder")
  save(out, "encoder", prg, state)

  export_decoder(model, out)
  dump_vocab(enc, out/"vocab.json")
  (out/"consts.json").write_text(json.dumps({
    "model_name": args.model, "is_multilingual": bool(model.is_multilingual), "n_vocab": enc.n_vocab,
    "samples_per_segment": SAMPLES_PER_SEGMENT, "max_tokens_to_sample": model.decoder.max_tokens_to_sample,
    "max_self_attn_cache_len": model.decoder.max_self_attn_cache_len,
    "sot": enc._special_tokens["<|startoftranscript|>"], "notimestamps": enc._special_tokens["<|notimestamps|>"],
    "eot": enc._special_tokens["<|endoftext|>"], "transcribe": enc._special_tokens["<|transcribe|>"],
    "langs": {k: enc._special_tokens[f"<|{k}|>"] for k in LANGUAGES.keys()} if model.is_multilingual else {},
  }, indent=2))
