from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='edge_generator',
    ext_modules=[
        CppExtension(
            'edge_generator',
            ['data/edge_generator.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)