from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension
)

setup(
    name='softmax_cuda',
    ext_modules=[
        CUDAExtension(
            name='softmax_cuda',
            sources=['softmax_cuda.cu'],
        )
    ],
    cmdclass={
        'build_ext':
            BuildExtension
    }
)