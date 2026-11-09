"""shapequant: data-free, training-free noise-shaped quantization of Hugging Face causal LMs."""
from .quantize import quantize_model, allocate
from .qlinear import load_quantized, QuantLinear
from .evaluate import perplexity, wikitext2_windows, windows_from_text
from .memory import estimate as estimate_memory

__version__ = "0.1.0"
__all__ = ["quantize_model", "allocate", "load_quantized", "QuantLinear", "perplexity", "wikitext2_windows",
           "windows_from_text", "estimate_memory"]
