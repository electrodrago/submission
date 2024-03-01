import os
import pip
import shutil
import subprocess

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

# Install requirement packages
install("openmim==0.3.9")
subprocess.call(["mim", "install", "mmcv>=2.0.0"])
install("mmagic==1.2.0")
install("mmengine==0.10.3")
install("diffusers==0.24.0")

# Setup models location
import mmagic
path = os.path.dirname(mmagic.__file__)

shutil.copytree("real_cleanvsr", os.path.join(path, "models/editors/real_cleanvsr"))
shutil.copytree("real_csrgan", os.path.join(path, "models/editors/real_csrgan"))
shutil.copy("editors/__init__.py", os.path.join(path, "models/editors/__init__.py"))
