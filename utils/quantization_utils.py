import torch
import warnings
from  torch.quantization import FakeQuantize
from torch.nn import Module

def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply


class LearnableFakeQuantize(FakeQuantize, Module):

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, dtype=torch.quint8, reduce_range=False):
        Module.__init__(self)
        self.reduce_range = reduce_range
        self.alpha = torch.nn.Parameter(torch.tensor([float('inf')]))
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.dtype = dtype
        self.qscheme = torch.per_tensor_symmetric
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.ch_axis = -1
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        self.round = TorchRound()
        if self.dtype == torch.qint8:
            if self.reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        else:
            if self.reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.debug_count = 0

    @torch.jit.export
    def enable_fake_quant(self, enabled=True):
        # type: (bool) -> None
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled=True):
        # type: (bool) -> None
        self.observer_enabled[0] = 1 if enabled else 0
        self.alpha.requires_grad = enabled

    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    @torch.jit.export
    def calculate_qparams(self):
        max_val = self.alpha.detach()
        min_val = -self.alpha.detach()
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "alpha should not be zero"
            )
            return torch.tensor([1.0]), torch.tensor([0])

        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64)
        device = self.alpha.device

        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(self.quant_max - self.quant_min) / 2)
        scale = torch.max(scale, self.eps)
        if self.dtype == torch.quint8:
            zero_point = zero_point.new_full(zero_point.size(), 128)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
        return scale, zero_point

    def forward(self, X):
        if self.alpha.data == float('inf'):
            self.alpha.data = torch.mean(torch.max(torch.max(torch.max(torch.abs(X),dim=1)[0],dim=1)[0],dim=1)[0]).unsqueeze(0)
        _scale, _zero_point = self.calculate_qparams()
        self.scale.resize_(_scale.shape)
        self.scale.copy_(_scale)
        self.zero_point.resize_(_zero_point.shape)
        self.zero_point.copy_(_zero_point)
        X = torch.max(torch.min(X, self.alpha), -self.alpha)
        if self.fake_quant_enabled:
            X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                        int(self.zero_point), self.quant_min,
                                                        self.quant_max)
        return X
