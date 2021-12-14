# Nvidia CUDA and cuDNN on Windows 11 WSL 2

### Sources

* https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
* https://developer.nvidia.com/rdp/cudnn-download
* https://forums.developer.nvidia.com/t/failure-to-install-cuda-on-wsl-2-ubuntu/128592/7

### CUDA installation

Install nvidia drivers on Windows system

`wsl --install -d Ubuntu`

```
$ nvidia-smi
Sun Dec  5 09:03:10 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.00       Driver Version: 510.06       CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T1200 La...  On   | 00000000:01:00.0 Off |                  N/A |
| N/A   50C    P8     3W /  N/A |    305MiB /  4096MiB |     N/A      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

```
$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.3 LTS
Release:        20.04
Codename:       focal
```

`$ sudo apt update`

`$ sudo apt dist-upgrade`

`$ sudo apt-get install build-essential cmake`

`$ mkdir ~/downloads`

`$ cd ~/downloads`

See https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local

(Do *not* install the .deb version or Windows driver bridge could be overriden)

`$ wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run`

`$ sudo sh cuda_11.5.1_495.29.05_linux.run`

accept, select Install and press ENTER

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ + [X] CUDA Toolkit 11.5                                                      │
│   [X] CUDA Samples 11.5                                                      │
│   [X] CUDA Demo Suite 11.5                                                   │
│   [X] CUDA Documentation 11.5                                                │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```

output: `***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 495.00 is required for CUDA 11.5 functionality to work.`

This means everything is fine.

`sudo sh cuda_11.5.1_495.29.05_linux.run --silent --driver` (install nvidia driver)

`$ nano ~/.bashrc` Add the following lines:

```
export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

`$ source ~/.bashrc`

### CUDA verfication

Copy cuda samples to a writeable dir,

`$ cp -r /usr/local/cuda-11.5/samples ~/cuda-11.5-samples`

`$ cd ~/cuda-11.5-samples/1_Utilities/deviceQuery`

`$ make`

```
$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA T1200 Laptop GPU"
  CUDA Driver Version / Runtime Version          11.6 / 11.5
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 4096 MBytes (4294705152 bytes)
  (016) Multiprocessors, (064) CUDA Cores/MP:    1024 CUDA Cores
  GPU Max Clock rate:                            1425 MHz (1.42 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.6, CUDA Runtime Version = 11.5, NumDevs = 1
Result = PASS
```

### cuDNN installation

* https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
* https://developer.nvidia.com/rdp/cudnn-download


Verify cuda version:
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

Download `Local Installer for Ubuntu20.04 x86_64 (Deb)` (required authentication)

Copy `cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_amd64.deb` to Ubuntu wsl2 `~/downloads`

`$ sudo apt install libfreeimage-dev`

`$ cd ~/downloads`

`$ sudo dpkg -i cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_amd64.deb`

`$ sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub`

`$ sudo apt-get install libcudnn8=8.3.1.22-1+cuda11.5`

`$ sudo apt-get install libcudnn8-dev=8.3.1.22-1+cuda11.5`

`$ sudo apt-get install libcudnn8-samples=8.3.1.22-1+cuda11.5`

### cuDNN verfication

Copy cuDNN samples to a writeable dir,

`$ cp -r /usr/src/cudnn_samples_v8/ $HOME`

`$ cd ~/cudnn_samples_v8/mnistCUDNN/`

`$ make clean && make`

```
$ ./mnistCUDNN | grep passed!
Test passed!
Test passed!
```
