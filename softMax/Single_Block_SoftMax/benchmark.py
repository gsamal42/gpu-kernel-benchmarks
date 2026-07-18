import torch
import triton
import triton.language as tl

print("before import")

import softmax_cuda

print("after import")

@triton.jit
def softmaxT(inPtr, outPtr, width, BLOCK_SIZE : tl.constexpr ):

    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    # load the ro
    row = tl.load (inPtr + offsets, mask=mask, other=-float("inf") )
    maxN = tl.max(row, axis=0)
    numerator = tl.exp (row - maxN)
    denominator = tl.sum (numerator, axis=0)
    result = numerator / denominator
    tl.store (outPtr + offsets , result, mask = mask )

x = torch.randn( 512, device='cuda')
y = softmax_cuda.forward(x)
torch.cuda.synchronize()
print(" cuda softmax o/p ",y)

outputT = torch.empty_like(x)
BLOCK_SIZE=512
grid = lambda meta : (triton.cdiv(512, meta['BLOCK_SIZE']),)
softmaxT[grid](x, outputT, 512, BLOCK_SIZE)
print ("Triton Softmax o/p", outputT)
