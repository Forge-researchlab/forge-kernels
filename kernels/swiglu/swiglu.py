import numbers

import torch
import triton
import triton.language as tl


MAX_ROW_BLOCK_SIZE = 65536


def _as_float(name: str, value: float) -> float:
    """Inputs: a scalar multiplier and its name. Outputs: a Python float. Logic: reject tensors/non-reals so multipliers stay compile-time constants."""
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Python scalar, not a tensor")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real Python scalar")
    return float(value)


def _calculate_settings(n_cols: int) -> tuple[int, int]:
    """Inputs: row width n_cols. Outputs: (block_size, num_warps). Logic: one row fits one block, so round up to a power of two and scale warps with width."""
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_ROW_BLOCK_SIZE:
        raise RuntimeError(
            f"SwiGLU hidden size {n_cols} requires block size {block_size}, "
            f"which exceeds the row-wise limit {MAX_ROW_BLOCK_SIZE}."
        )

    num_warps = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


def _check_same_shape_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    """Inputs: gate/up tensors. Outputs: none (raises on mismatch). Logic: enforce matching shape/device/dtype and floating point before launch."""
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    if gate.device != up.device:
        raise ValueError("gate and up must be on the same device")
    if gate.dtype != up.dtype:
        raise TypeError(f"gate and up must have the same dtype, got {gate.dtype} and {up.dtype}")
    if not gate.is_floating_point() or not up.is_floating_point():
        raise TypeError("gate and up must be floating point tensors")


def _check_cuda_dtype(x: torch.Tensor) -> None:
    """Inputs: a tensor. Outputs: none (raises on unsupported dtype). Logic: the CUDA kernels only handle fp16/bf16/fp32."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"CUDA SwiGLU supports fp16, bf16, and fp32, got {x.dtype}")


def _check_packed_input(gate_up: torch.Tensor) -> None:
    """Inputs: a packed gate_up tensor. Outputs: none (raises on bad input). Logic: the last dim splits into gate||up halves, so it must be even and floating point."""
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"packed gate_up last dimension must be even, got {gate_up.shape[-1]}")
    if not gate_up.is_floating_point():
        raise TypeError("gate_up must be a floating point tensor")


@triton.jit
def _swiglu_forward_kernel(
    out_ptr,
    gate_ptr,
    up_ptr,
    n_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row computes out = silu(gate * gate_mult) * up * down_mult.

    The SiLU is evaluated in fp32 for numerical stability and cast back to the
    input dtype before the elementwise product, matching the PyTorch reference.
    """
    row_idx = tl.program_id(0).to(tl.int64)  # int64 to keep row_start in range for tall tensors
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols  # guard the power-of-two tail past the real row width
    row_start = row_idx * n_cols

    gate = tl.load(gate_ptr + row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32) * gate_multiplier
    up = tl.load(up_ptr + row_start + offsets, mask=mask, other=0.0)

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)  # SiLU(x) = x * sigmoid(x)
    out = (silu_gate.to(gate_dtype) * up) * down_multiplier
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dgate_ptr,
    dup_ptr,
    dout_ptr,
    gate_ptr,
    up_ptr,
    n_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row computes the SwiGLU gradients w.r.t. gate and up.

    With s = silu(gate * gate_mult), out = s * up * down_mult, so:
      dup   = dout * down_mult * s
      dgate = dout * down_mult * up * silu'(gate * gate_mult) * gate_mult
    where silu'(x) = silu(x) * (1 - sigmoid(x)) + sigmoid(x).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_idx * n_cols

    dout = tl.load(dout_ptr + row_start + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    gate_fp32 = gate.to(tl.float32) * gate_multiplier
    up = tl.load(up_ptr + row_start + offsets, mask=mask, other=0.0)

    sig = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sig
    dout_scaled = dout * down_multiplier

    dup = dout_scaled * silu_gate.to(gate_dtype)
    # (silu_gate * (1 - sig) + sig) is silu'(gate_fp32); trailing gate_multiplier is the chain rule.
    dgate = dout_scaled * up * (silu_gate * (1.0 - sig) + sig) * gate_multiplier

    tl.store(dgate_ptr + row_start + offsets, dgate, mask=mask)
    tl.store(dup_ptr + row_start + offsets, dup, mask=mask)


@triton.jit
def _swiglu_packed_forward_kernel(
    out_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU forward over a packed [gate || up] row of width total_cols = 2 * n_cols.

    gate is the first n_cols of the input row, up the second; the output row holds
    only the n_cols result, so input and output use different row strides.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row_start = row_idx * total_cols   # packed row: gate||up
    output_row_start = row_idx * n_cols      # result row: n_cols wide

    gate = tl.load(gate_up_ptr + input_row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    up = tl.load(gate_up_ptr + input_row_start + n_cols + offsets, mask=mask, other=0.0)  # up half
    gate_fp32 = gate.to(tl.float32) * gate_multiplier

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)
    out = (silu_gate.to(gate_dtype) * up) * down_multiplier
    tl.store(out_ptr + output_row_start + offsets, out, mask=mask)


@triton.jit
def _swiglu_packed_backward_kernel(
    dgate_up_ptr,
    dout_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    total_cols: tl.constexpr,
    gate_multiplier: tl.constexpr,
    down_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU backward for the packed layout; same math as the unpacked backward.

    Reads dout (n_cols wide) plus the gate/up halves of the packed row, then
    writes dgate and dup back into the gate and up slots of dgate_up_ptr in place.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row_start = row_idx * total_cols
    output_row_start = row_idx * n_cols

    dout = tl.load(dout_ptr + output_row_start + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_up_ptr + input_row_start + offsets, mask=mask, other=0.0)
    gate_dtype = gate.dtype
    up = tl.load(gate_up_ptr + input_row_start + n_cols + offsets, mask=mask, other=0.0)
    gate_fp32 = gate.to(tl.float32) * gate_multiplier

    sig = tl.sigmoid(gate_fp32)
    silu_gate = gate_fp32 * sig
    dout_scaled = dout * down_multiplier

    dup = dout_scaled * silu_gate.to(gate_dtype)
    # (silu_gate * (1 - sig) + sig) is silu'(gate_fp32); trailing gate_multiplier is the chain rule.
    dgate = dout_scaled * up * (silu_gate * (1.0 - sig) + sig) * gate_multiplier

    # Write grads back into the matching gate/up slots of the packed buffer.
    tl.store(dgate_up_ptr + input_row_start + offsets, dgate, mask=mask)
    tl.store(dgate_up_ptr + input_row_start + n_cols + offsets, dup, mask=mask)


def torch_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    """Inputs: gate/up tensors and multipliers. Outputs: SwiGLU result. Logic: pure-PyTorch reference used for correctness checks and CPU fallback."""
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    silu_gate = torch.nn.functional.silu(gate.float() * gate_multiplier).to(dtype=gate.dtype)
    return (silu_gate * up) * down_multiplier


def torch_swiglu_packed_reference(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    """Inputs: packed gate_up tensor and multipliers. Outputs: SwiGLU result. Logic: split the last dim into gate/up halves and defer to the reference."""
    _check_packed_input(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return torch_swiglu_reference(gate, up, gate_multiplier, down_multiplier)


def swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
):
    """Inputs: gate/up tensors and multipliers. Outputs: (result, gate_2d, up_2d). Logic: flatten to rows, launch one program per row, return the flattened views for backward to reuse."""
    _check_same_shape_inputs(gate, up)
    _check_cuda_dtype(gate)
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)

    original_shape = gate.shape
    n_cols = original_shape[-1]
    gate_2d = gate.contiguous().view(-1, n_cols)
    up_2d = up.contiguous().view(-1, n_cols)
    out = torch.empty_like(gate_2d)
    block_size, num_warps = _calculate_settings(n_cols)

    _swiglu_forward_kernel[(gate_2d.shape[0],)](
        out,
        gate_2d,
        up_2d,
        n_cols=n_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.view(original_shape), gate_2d, up_2d


def swiglu_backward(
    dout: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
):
    """Inputs: dout, saved gate/up, multipliers, preserve flag. Outputs: (dgate, dup). Logic: by default overwrite gate/up in place to save memory; preserve_inputs keeps them by writing to fresh buffers."""
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    original_shape = dout.shape
    n_cols = original_shape[-1]
    dout_2d = dout.contiguous().view(-1, n_cols)
    block_size, num_warps = _calculate_settings(n_cols)

    # In place by default; preserve_inputs trades memory for keeping gate/up intact.
    dgate = torch.empty_like(gate) if preserve_inputs else gate
    dup = torch.empty_like(up) if preserve_inputs else up

    _swiglu_backward_kernel[(dout_2d.shape[0],)](
        dgate,
        dup,
        dout_2d,
        gate,
        up,
        n_cols=n_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dgate.view(original_shape), dup.view(original_shape)


def swiglu_packed_forward(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
):
    """Inputs: packed gate_up tensor and multipliers. Outputs: (result, gate_up_2d). Logic: result is half the width of the packed input; return the flattened packed view for backward."""
    _check_packed_input(gate_up)
    _check_cuda_dtype(gate_up)
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)

    original_shape = gate_up.shape
    total_cols = original_shape[-1]
    n_cols = total_cols // 2
    gate_up_2d = gate_up.contiguous().view(-1, total_cols)
    out = torch.empty((gate_up_2d.shape[0], n_cols), device=gate_up.device, dtype=gate_up.dtype)
    block_size, num_warps = _calculate_settings(n_cols)

    _swiglu_packed_forward_kernel[(gate_up_2d.shape[0],)](
        out,
        gate_up_2d,
        n_cols=n_cols,
        total_cols=total_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.view(*original_shape[:-1], n_cols), gate_up_2d


def swiglu_packed_backward(
    dout: torch.Tensor,
    gate_up: torch.Tensor,
    input_shape: torch.Size,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
):
    """Inputs: dout, saved packed gate_up, original input_shape, multipliers, preserve flag. Outputs: dgate_up packed like the input. Logic: kernel writes both grads back into the packed buffer (in place unless preserve_inputs)."""
    gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
    down_multiplier = _as_float("down_multiplier", down_multiplier)
    n_cols = dout.shape[-1]
    total_cols = input_shape[-1]
    dout_2d = dout.contiguous().view(-1, n_cols)
    block_size, num_warps = _calculate_settings(n_cols)

    dgate_up = torch.empty_like(gate_up) if preserve_inputs else gate_up

    _swiglu_packed_backward_kernel[(dout_2d.shape[0],)](
        dgate_up,
        dout_2d,
        gate_up,
        n_cols=n_cols,
        total_cols=total_cols,
        gate_multiplier=gate_multiplier,
        down_multiplier=down_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dgate_up.view(input_shape)


class ForgeSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for the unpacked SwiGLU: saves the flattened gate/up for backward."""

    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        gate_multiplier: float = 1.0,
        down_multiplier: float = 1.0,
        preserve_inputs: bool = False,
    ):
        """Run the forward kernel and stash gate/up plus the scalar multipliers on ctx for backward."""
        y, gate_2d, up_2d = swiglu_forward(gate, up, gate_multiplier, down_multiplier)
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
        ctx.down_multiplier = _as_float("down_multiplier", down_multiplier)
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """Return grads for (gate, up); the three trailing None match the non-tensor forward args."""
        gate, up = ctx.saved_tensors
        dgate, dup = swiglu_backward(
            dout,
            gate,
            up,
            ctx.gate_multiplier,
            ctx.down_multiplier,
            ctx.preserve_inputs,
        )
        return dgate, dup, None, None, None


class ForgePackedSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for the packed SwiGLU: also records the input shape so backward can repack."""

    @staticmethod
    def forward(
        ctx,
        gate_up: torch.Tensor,
        gate_multiplier: float = 1.0,
        down_multiplier: float = 1.0,
        preserve_inputs: bool = False,
    ):
        """Run the packed forward kernel and save the flattened gate_up, its shape, and multipliers for backward."""
        y, gate_up_2d = swiglu_packed_forward(gate_up, gate_multiplier, down_multiplier)
        ctx.save_for_backward(gate_up_2d)
        ctx.input_shape = gate_up.shape
        ctx.gate_multiplier = _as_float("gate_multiplier", gate_multiplier)
        ctx.down_multiplier = _as_float("down_multiplier", down_multiplier)
        ctx.preserve_inputs = bool(preserve_inputs)
        return y

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """Return the packed grad for gate_up; the three trailing None match the non-tensor forward args."""
        (gate_up,) = ctx.saved_tensors
        dgate_up = swiglu_packed_backward(
            dout,
            gate_up,
            ctx.input_shape,
            ctx.gate_multiplier,
            ctx.down_multiplier,
            ctx.preserve_inputs,
        )
        return dgate_up, None, None, None


def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
) -> torch.Tensor:
    """Inputs: gate/up tensors, multipliers, preserve flag. Outputs: SwiGLU result. Logic: public entry point; CPU tensors take the PyTorch reference, CUDA tensors take the autograd-enabled Triton path."""
    _check_same_shape_inputs(gate, up)
    if not gate.is_cuda:
        return torch_swiglu_reference(gate, up, gate_multiplier, down_multiplier)
    return ForgeSwiGLUFunction.apply(gate, up, gate_multiplier, down_multiplier, preserve_inputs)


def swiglu_packed(
    gate_up: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
    preserve_inputs: bool = False,
) -> torch.Tensor:
    """Inputs: packed gate_up tensor, multipliers, preserve flag. Outputs: SwiGLU result (half width). Logic: public entry point for the packed layout; CPU falls back to the reference, CUDA uses the Triton path."""
    _check_packed_input(gate_up)
    if not gate_up.is_cuda:
        return torch_swiglu_packed_reference(gate_up, gate_multiplier, down_multiplier)
    return ForgePackedSwiGLUFunction.apply(gate_up, gate_multiplier, down_multiplier, preserve_inputs)
