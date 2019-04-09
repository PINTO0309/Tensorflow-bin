# Tensorflow-bin
Prebuilt binary with Tensorflow Lite enabled. For RaspberryPi / Jetson TX2.  
And, The following problem was solved. **[#15062](https://github.com/tensorflow/tensorflow/issues/15062), [#21574](https://github.com/tensorflow/tensorflow/issues/21574), [#21855](https://github.com/tensorflow/tensorflow/issues/21855),  [#23082](https://github.com/tensorflow/tensorflow/issues/23082), [#25120](https://github.com/tensorflow/tensorflow/issues/25120), [#25748](https://github.com/tensorflow/tensorflow/pull/25748)**  
  
1. `undefined symbol: _ZN6tflite12tensor_utils39NeonMatrixBatchVectorMultiplyAccumulateEPKaiiS2_PKfiPfi`  
2. `Bus Error`
  
RaspberryPi3 ... armv7l  
Jetson TX2 ... aarch64  
  
Bazel's pre-build binay is below.  
**https://github.com/PINTO0309/Bazel_bin.git**  

Cross compilation recommends using **`lhelontra`** repository.  
**https://github.com/lhelontra/tensorflow-on-arm.git**

## Binary type
**`Best`** is the highest performance binary when using **`Python API`**.  
  
**Python 2.x**　Be sure to rename to **`tensorflow-1.11.0-cp27-cp27mu-linux_armv7l.whl`** and use the pip2 command.

|Best|.whl|jemalloc|MPI|4Threads|
|:--:|:--|:--:|:--:|:--:|
||tensorflow-1.11.0-cp27-cp27mu-linux_armv7l.whl||||
|:star:|tensorflow-1.11.0-cp27-cp27mu-linux_armv7l_jemalloc.whl|○|||
||tensorflow-1.11.0-cp27-cp27mu-linux_armv7l_jemalloc_mpi_multithread.whl|○|○|○|

**Python 3.x**　Be sure to rename to **`tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl`** and use the pip3 command.

|Best|.whl|jemalloc|MPI|4Threads|
|:--:|:--|:--:|:--:|:--:|
||tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl||||
||tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc.whl|○|||
||tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc_mpi.whl|○|○||
||tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc_mpi_multithread.whl|○|○|○|
|:star:|tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc_multithread.whl|○||○|

**Python 3.x + Tensorflow v1.13.1**  

|Best|.whl|jemalloc|MPI|4Threads|Note|
|:--:|:--|:--:|:--:|:--:|:--|
|:star:|tensorflow-1.13.1-cp35-cp35m-linux_armv7l.whl|○||○||

**Python 3.x + Tensorflow v2**  

|Best|.whl|jemalloc|MPI|4Threads|Note|
|:--:|:--|:--:|:--:|:--:|:--|
|:star:|tensorflow-2.0.0a0-cp35-cp35m-linux_armv7l.whl|○||○|Alpha version|

## Usage
**Example of Python 2.x series**
```bash
$ sudo apt-get install python-pip python3-pip python-scipy libhdf5-dev
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo pip2 uninstall tensorflow
$ wget -O tensorflow-1.11.0-cp27-cp27mu-linux_armv7l.whl https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.11.0-cp27-cp27mu-linux_armv7l_jemalloc.whl
$ sudo pip2 install tensorflow-1.11.0-cp27-cp27mu-linux_armv7l.whl

【Required】 Restart the terminal.
```

**Example of Python 3.x series**
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo pip3 uninstall tensorflow
$ wget -O tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.11.0-cp35-cp35m-linux_armv7l_jemalloc_multithread.whl
$ sudo pip3 install tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl

【Required】 Restart the terminal.
```
**Example of Python 3.x + Tensorflow v2 series**
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock
$ sudo apt update;sudo apt upgrade
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-2.0.0a0-cp35-cp35m-linux_armv7l.whl
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.0.0a0-cp35-cp35m-linux_armv7l.whl

【Required】 Restart the terminal.
```

## Operation check
**Example of Python 2.x series**
```bash
$ python
>>> import tensorflow
>>> tensorflow.__version__
1.11.0
>>> exit()
```

**Example of Python 3.x series**
```bash
$ python3
>>> import tensorflow
>>> tensorflow.__version__
1.11.0
>>> exit()
```

**Sample of MultiThread x4**
- Preparation of test environment
```bash
$ cd ~;mkdir test
$ curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > ~/test/grace_hopper.bmp
$ curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz | tar xzv -C ~/test mobilenet_v1_1.0_224/labels.txt
$ mv ~/test/mobilenet_v1_1.0_224/labels.txt ~/test/
$ curl http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz | tar xzv -C ~/test
$ cp tensorflow/tensorflow/contrib/lite/examples/python/label_image.py ~/test
```
<details><summary>[Sample Code] label_image.py</summary><div>

```python
import argparse
import numpy as np
import time

from PIL import Image

# Tensorflow -v1.12.0
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

# Tensorflow v1.13.0+, v2.x.x
#from tensorflow.lite.python import interpreter as interpreter_wrapper

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels
if __name__ == "__main__":
  floating_model = False
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", default="/tmp/grace_hopper.bmp", \
    help="image to be classified")
  parser.add_argument("-m", "--model_file", \
    default="/tmp/mobilenet_v1_1.0_224_quant.tflite", \
    help=".tflite model to be executed")
  parser.add_argument("-l", "--label_file", default="/tmp/labels.txt", \
    help="name of file containing labels")
  parser.add_argument("--input_mean", default=127.5, help="input_mean")
  parser.add_argument("--input_std", default=127.5, \
    help="input standard deviation")
  parser.add_argument("--num_threads", default=1, help="number of threads")
  args = parser.parse_args()

  interpreter = interpreter_wrapper.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # check the type of the input tensor
  if input_details[0]['dtype'] == np.float32:
    floating_model = True
  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image)
  img = img.resize((width, height))
  # add N dim
  input_data = np.expand_dims(img, axis=0)
  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_num_threads(int(args.num_threads))
  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)
  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  for i in top_k:
    if floating_model:
      print('{0:08.6f}'.format(float(results[i]))+":", labels[i])
    else:
      print('{0:08.6f}'.format(float(results[i]/255.0))+":", labels[i])

  print("time: ", stop_time - start_time)
```

</div></details>
<br>

- Run test
```bash
$ cd ~/test
$ python3 label_image.py \
--num_threads 1 \
--image grace_hopper.bmp \
--model_file mobilenet_v1_1.0_224_quant.tflite \
--label_file labels.txt

0.415686: 653:military uniform
0.352941: 907:Windsor tie
0.058824: 668:mortarboard
0.035294: 458:bow tie, bow-tie, bowtie
0.035294: 835:suit, suit of clothes
time:  0.4152982234954834
```
```bash
$ cd ~/test
$ python3 label_image.py \
--num_threads 4 \
--image grace_hopper.bmp \
--model_file mobilenet_v1_1.0_224_quant.tflite \
--label_file labels.txt

0.415686: 653:military uniform
0.352941: 907:Windsor tie
0.058824: 668:mortarboard
0.035294: 458:bow tie, bow-tie, bowtie
0.035294: 835:suit, suit of clothes
time:  0.1647195816040039
```

## Build Parameter
  
============================================================  
  
**Tensorflow v1.11.0**  

============================================================  
  
**Python2.x - Bazel 0.17.2**
```bash
$ sudo apt-get install -y openmpi-bin libopenmpi-dev libhdf5-dev

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
$ ./configure

Please specify the location of python. [Default is /usr/bin/python]: 


Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/local/lib
  /home/pi/tensorflow/tensorflow/contrib/lite/tools/make/gen/rpi_armv7l/lib
  /usr/lib/python2.7/dist-packages
  /opt/movidius/caffe/python
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
No jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
No Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]: n
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
```
```bash
$ sudo bazel build --config opt --local_resources 1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```
```bash
$ sudo ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip2 install /tmp/tensorflow_pkg/tensorflow-1.11.0-cp27-cp27mu-linux_armv7l.whl
```

**Python3.x- Bazel 0.17.2 + ZRAM + PythonAPI(MultiThread) Feb 23, 2019, Compilation work completed**
```bash
$ sudo nano /etc/dphys-swapfile
CONF_SWAPFILE=2048
CONF_MAXSWAP=2048

$ sudo systemctl stop dphys-swapfile
$ sudo systemctl start dphys-swapfile

$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/zram.sh
$ chmod 755 zram.sh
$ sudo mv zram.sh /etc/init.d/
$ sudo update-rc.d zram.sh defaults
$ sudo reboot

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
```
Modify the program with reference to the following.  
<details><summary>tensorflow/contrib/lite/examples/python/label_image.py</summary><div>

```python
import argparse
import numpy as np
import time

from PIL import Image

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels
if __name__ == "__main__":
  floating_model = False
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--image", default="/tmp/grace_hopper.bmp", \
    help="image to be classified")
  parser.add_argument("-m", "--model_file", \
    default="/tmp/mobilenet_v1_1.0_224_quant.tflite", \
    help=".tflite model to be executed")
  parser.add_argument("-l", "--label_file", default="/tmp/labels.txt", \
    help="name of file containing labels")
  parser.add_argument("--input_mean", default=127.5, help="input_mean")
  parser.add_argument("--input_std", default=127.5, \
    help="input standard deviation")
  parser.add_argument("--num_threads", default=1, help="number of threads")
  args = parser.parse_args()

  interpreter = interpreter_wrapper.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # check the type of the input tensor
  if input_details[0]['dtype'] == np.float32:
    floating_model = True
  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image)
  img = img.resize((width, height))
  # add N dim
  input_data = np.expand_dims(img, axis=0)
  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_num_threads(int(args.num_threads))
  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)
  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  for i in top_k:
    if floating_model:
      print('{0:08.6f}'.format(float(results[i]))+":", labels[i])
    else:
      print('{0:08.6f}'.format(float(results[i]/255.0))+":", labels[i])

  print("time: ", stop_time - start_time)
```

</div></details>

<details><summary>tensorflow/contrib/lite/python/interpreter.py</summary><div>

```python
#Add the following two lines to the last line

  def set_num_threads(self, i):
    return self._interpreter.SetNumThreads(i)
```

</div></details>

<details><summary>tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.cc</summary><div>

```cpp
//Corrected the vicinity of the last line as follows

  PyObject* InterpreterWrapper::ResetVariableTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::SetNumThreads(int i) {
  interpreter_->SetNumThreads(i);
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
```

</div></details>

<details><summary>tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.h</summary><div>

```cpp
//Modified the middle of the logic as follows

  // should be the interpreter object providing the memory.
  PyObject* tensor(PyObject* base_object, int i);

  PyObject* SetNumThreads(int i);

 private:
  // Helper function to construct an `InterpreterWrapper` object.
  // It only returns InterpreterWrapper if it can construct an `Interpreter`.
```

</div></details>
<br>

```bash
$ ./configure

Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
  /opt/movidius/caffe/python
Please input the desired Python library path to use.  Default is [/usr/local/lib] /usr/local/lib/python3.5/dist-packages

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
No jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
No Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]: n
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```
```bash
$ sudo bazel build --config opt --local_resources 1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```
```bash
$ sudo -s
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# exit
$ sudo pip3 install /tmp/tensorflow_pkg/tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
```

**Python3.x + jemalloc + MPI + MultiThread [C++ Only]**  
  
Edit **`tensorflow/tensorflow/contrib/mpi/mpi_rendezvous_mgr.cc`** Line139 / Line140, Line261.
```cxx
  MPIRendezvousMgr* mgr =
      reinterpret_cast<MPIRendezvousMgr*>(this->rendezvous_mgr_);
- mgr->QueueRequest(parsed.FullKey().ToString(), step_id_,
-                   std::move(request_call), rendezvous_call);
+ mgr->QueueRequest(string(parsed.FullKey()), step_id_, std::move(request_call),
+                   rendezvous_call);
}
 MPIRemoteRendezvous::~MPIRemoteRendezvous() {}


        std::function<MPISendTensorCall*()> res = std::bind(
            send_cb, status, send_args, recv_args, val, is_dead, mpi_send_call);
-       SendQueueEntry req(parsed.FullKey().ToString().c_str(), std::move(res));
+       SendQueueEntry req(string(parsed.FullKey()), std::move(res));
         this->QueueSendRequest(req);
```
Edit **`tensorflow/tensorflow/contrib/mpi/mpi_rendezvous_mgr.h`** Line74
```cxx
  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id,
            const bool is_dead) {
-   mRes_.set_key(parsed.FullKey().ToString());
+   mRes_.set_key(string(parsed.FullKey()));
    mRes_.set_step_id(step_id);
    mRes_.mutable_response()->set_is_dead(is_dead);
    mRes_.mutable_response()->set_send_start_micros(
```
Edit **`tensorflow/tensorflow/contrib/lite/interpreter.cc`** Line127.
```cxx
-  context_.recommended_num_threads = -1;
+  context_.recommended_num_threads = 4;
```
```bash
$ sudo apt-get install -y libhdf5-dev
$ sudo pip3 install keras_applications==1.0.4 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.2 --no-deps
$ sudo pip3 install h5py==2.8.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
$ ./configure

Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
  /opt/movidius/caffe/python
Please input the desired Python library path to use.  Default is [/usr/local/lib] /usr/local/lib/python3.5/dist-packages

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for Tensorflow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
No Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]: n
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: y
MPI support will be enabled for Tensorflow.

Please specify the MPI toolkit folder. [Default is /usr]: /usr/lib/arm-linux-gnueabihf/openmpi

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```
```bash
$ sudo bazel build --config opt --local_resources 1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```

**Python3.x + jemalloc + XLA JIT (Build impossible)**  
  
```bash
$ sudo apt-get install -y libhdf5-dev
$ sudo pip3 install keras_applications==1.0.4 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.2 --no-deps
$ sudo pip3 install h5py==2.8.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ JAVA_OPTIONS=-Xmx256M

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
$ ./configure

Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
  /opt/movidius/caffe/python
Please input the desired Python library path to use.  Default is [/usr/local/lib] /usr/local/lib/python3.5/dist-packages

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for Tensorflow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: n
No Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: y
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with nGraph support? [y/N]: n
No nGraph support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
MPI support will be enabled for Tensorflow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```
```bash
$ sudo bazel build --config opt --local_resources 1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```

**Python3.x + TX2 aarch64 - Bazel 0.18.1 (JetPack-L4T-3.3-linux-x64_b39)**

```
- L4T R28.2.1（TX2 / TX2i）
- L4T R28.2（TX1）
- CUDA 9.0
- cuDNN 7.1.5
- TensorRT 4.0
- VisionWorks 1.6
```

**https://github.com/tensorflow/tensorflow/issues/21574#issuecomment-429758923**  
**https://github.com/tensorflow/serving/issues/832**  
**https://docs.nvidia.com/deeplearning/sdk/nccl-archived/nccl_2213/nccl-install-guide/index.html**  

```
build --action_env PYTHON_BIN_PATH="/usr/bin/python3"
build --action_env PYTHON_LIB_PATH="/usr/local/lib/python3.5/dist-packages"
build --python_path="/usr/bin/python3"
build --define with_jemalloc=true
build:gcp --define with_gcp_support=true
build:hdfs --define with_hdfs_support=true
build:aws --define with_aws_support=true
build:kafka --define with_kafka_support=true
build:xla --define with_xla_support=true
build:gdr --define with_gdr_support=true
build:verbs --define with_verbs_support=true
build:ngraph --define with_ngraph_support=true
build --action_env TF_NEED_OPENCL_SYCL="0"
build --action_env TF_NEED_CUDA="1"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-9.0"
build --action_env TF_CUDA_VERSION="9.0"
build --action_env CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
build --action_env TF_CUDNN_VERSION="7"
build --action_env NCCL_INSTALL_PATH="/usr/local"
build --action_env TF_NCCL_VERSION="2"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="3.5,7.0"
build --action_env LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:../src/.libs"
build --action_env TF_CUDA_CLANG="0"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
build --config=cuda
test --config=cuda
build --define grpc_no_ares=true
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
```
  
```bash
$ sudo apt-get install -y libhdf5-dev
$ sudo pip3 install keras_applications==1.0.4 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.2 --no-deps
$ sudo pip3 install h5py==2.8.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ bazel build -c opt --config=cuda --local_resources 3072.0,4.0,1.0 --verbose_failures //tensorflow/tools/pip_package:build_pip_package
```
  
============================================================  
  
**Tensorflow v1.12.0**  

============================================================  
  
**Python3.x - Bazel 0.18.1 (Nov 15, 2018 Under construction)**  
  
```bash
$ sudo nano /etc/dphys-swapfile
CONF_SWAPFILE=2048
CONF_MAXSWAP=2048

$ sudo systemctl stop dphys-swapfile
$ sudo systemctl start dphys-swapfile

$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/zram.sh
$ chmod 755 zram.sh
$ sudo mv zram.sh /etc/init.d/
$ sudo update-rc.d zram.sh defaults
$ sudo reboot

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow

$ ./configure
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.18.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
  /opt/movidius/caffe/python
Please input the desired Python library path to use.  Default is [/usr/local/lib]
/usr/local/lib/python3.5/dist-packages
Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: n
No Apache Ignite support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
Configuration finished
```
~~https://github.com/tensorflow/tensorflow/issues/22819~~  
~~https://github.com/tensorflow/tensorflow/commit/d80eb525e94763e09cbb9fa3cbef9a0f64e2cb2a~~  
~~https://github.com/tensorflow/tensorflow/commit/5847293aeb9ab45a02c4231c40569a15bd4541c6~~  
https://github.com/tensorflow/tensorflow/issues/23721  
https://github.com/tensorflow/tensorflow/pull/25748  
https://github.com/tensorflow/tensorflow/issues/25120#issuecomment-464296755  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package  
```
$ sudo bazel --host_jvm_args=-Xmx512m build \
--config=opt \
--local_resources=1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```
  
============================================================  
  
**Tensorflow v2.0.0-alpha - Bazel 0.19.2**  

============================================================  

```bash
$ sudo nano /etc/dphys-swapfile
CONF_SWAPFILE=2048
CONF_MAXSWAP=2048

$ sudo systemctl stop dphys-swapfile
$ sudo systemctl start dphys-swapfile

$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/zram.sh
$ chmod 755 zram.sh
$ sudo mv zram.sh /etc/init.d/
$ sudo update-rc.d zram.sh defaults
$ sudo reboot

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock
$ sudo apt update;sudo apt upgrade

$ cd ~
$ git clone https://github.com/PINTO0309/Bazel_bin.git
$ cd Bazel_bin
$ ./0.19.2/Raspbian_armhf/install.sh

$ cd ~
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
```
- tensorflow/lite/python/interpreter.py
```bash
# Add the following two lines to the last line
  def set_num_threads(self, i):
    return self._interpreter.SetNumThreads(i)
```
- tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc
```C++
// Corrected the vicinity of the last line as follows
PyObject* InterpreterWrapper::ResetVariableTensors() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::SetNumThreads(int i) {
  interpreter_->SetNumThreads(i);
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
```
- tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h
```C++
  // should be the interpreter object providing the memory.
  PyObject* tensor(PyObject* base_object, int i);

  PyObject* SetNumThreads(int i);

 private:
  // Helper function to construct an `InterpreterWrapper` object.
  // It only returns InterpreterWrapper if it can construct an `Interpreter`.
```
- tensorflow/tensorflow/core/kernels/BUILD
```python
cc_library(
    name = "linalg",
    deps = [
        ":cholesky_grad",
        ":cholesky_op",
        ":determinant_op",
        ":lu_op",
        ":matrix_exponential_op",
        ":matrix_inverse_op",
        ":matrix_logarithm_op",
        ":matrix_solve_ls_op",
        ":matrix_solve_op",
        ":matrix_triangular_solve_op",
        ":qr_op",
        ":self_adjoint_eig_op",
        ":self_adjoint_eig_v2_op",
        ":svd_op",
        ":tridiagonal_solve_op",
    ],
)
```
- tensorflow/tensorflow/core/kernels/BUILD - Delete the following
```python
tf_kernel_library(
    name = "matrix_square_root_op",
    prefix = "matrix_square_root_op",
    deps = LINALG_DEPS,
)
```
- tensorflow/lite/tools/make/Makefile
```python
BUILD_WITH_NNAPI=false
```
- configure
```bash
$ ./configure 
Extracting Bazel installation...
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.19.2- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /home/b920405/git/caffe-jacinto/python
  /opt/intel//computer_vision_sdk_2018.5.455/python/python3.5/ubuntu16
  /opt/intel//computer_vision_sdk_2018.5.455/python/python3.5
  .
  /opt/intel//computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer
  /opt/movidius/caffe/python
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.5/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib]
/usr/local/lib/python3.5/dist-packages
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
    --config=gdr            # Build with GDR support.
    --config=verbs          # Build with libverbs support.
    --config=ngraph         # Build with Intel nGraph support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=noaws          # Disable AWS S3 filesystem support.
    --config=nogcp          # Disable GCP support.
    --config=nohdfs         # Disable HDFS support.
    --config=noignite       # Disable Apache Ignite support.
    --config=nokafka        # Disable Apache Kafka support.
    --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```
- build
```bash
$ sudo bazel --host_jvm_args=-Xmx512m build \
--config=opt \
--config=noaws \
--config=nogcp \
--config=nohdfs \
--config=noignite \
--config=nokafka \
--config=nonccl \
--local_resources=1024.0,0.5,0.5 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
//tensorflow/tools/pip_package:build_pip_package
```
```bash
$ su --preserve-environment
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# exit
$ sudo cp /tmp/tensorflow_pkg/tensorflow-2.0.0a0-cp35-cp35m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.0.0a0-cp35-cp35m-linux_armv7l.whl 
```
