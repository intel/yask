import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install

class Install(install):
    """Customized setuptools install command - builds protos on install."""
    def run(self):
        protoc_command = ['make', 'compiler']
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)

        protoc_command = ['make', 'compiler-api']
        if subprocess.call(protoc_command) != 0:
            sys.exit(-1)

        install.run(self)

setup(
    name='yask',
    version='v2-alpha',
    description='YASK--Yet Another Stencil Kernel: '
    'A framework to facilitate exploration of the HPC '
    'stencil-performance design space',
    url='https://01.org/yask',
    author='Intel Corporation',
    license='MIT',
    package_dir = {'': 'lib'},
    cmdclass={
        'install': Install,
    }
)
