# Nvidia accelerated TensorFlow on Windows 11 wsl2

### Sources

* https://www.tensorflow.org/install/gpu
* https://www.anaconda.com/products/individual

### Install Anaconda Python

`$ nano ~/.bashrc`

Replace export LD_LIBRARY_PATH=... with `export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:/usr/local/cuda/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

`$ source ~/.bashrc`

`$ cd ~/downloads`

`$ wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh`

`$ sh Anaconda3-2021.11-Linux-x86_64.sh`

`$ source ~/.bashrc`

### Install TensorFlow

`(base) $ pip install --upgrade pip`

`(base) $ pip install tensorflow`

### Verify TensorFlow GPU acceleration

`$ python <<< $'import tensorflow as tf\nprint(len(tf.config.list_physical_devices(\'GPU\')))'`

```
2021-12-05 21:19:20.791569: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2021-12-05 21:19:20.821226: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
2021-12-05 21:19:20.822066: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
1
```

The last number in the output shows number of gpu's available to TensorFlow.

'Your kernel may have been built without NUMA support.' is probably normal under WSL 2.

Happy coding!