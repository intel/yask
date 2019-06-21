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
    version='v3',
    description='YASK--Yet Another Stencil Kit',
    url='https://github.com/intel/yask',
    author='Intel Corporation',
    license='MIT',
    packages = ['yask'],
    cmdclass={
        'install': Install,
    }
)
