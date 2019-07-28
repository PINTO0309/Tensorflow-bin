# Tensorflow-bin
  
**Older versions of Wheel files can be obtained from the [Fork destination repository](https://github.com/PINTO0309/Tensorflow-bin/network/members).**  
  
Prebuilt binary with Tensorflow Lite enabled. For RaspberryPi.  
And, The following problem was solved. **[#15062](https://github.com/tensorflow/tensorflow/issues/15062), [#21574](https://github.com/tensorflow/tensorflow/issues/21574), [#21855](https://github.com/tensorflow/tensorflow/issues/21855),  [#23082](https://github.com/tensorflow/tensorflow/issues/23082), [#25120](https://github.com/tensorflow/tensorflow/issues/25120), [#25748](https://github.com/tensorflow/tensorflow/pull/25748), [#29617](https://github.com/tensorflow/tensorflow/issues/29617), [#30359](https://github.com/tensorflow/tensorflow/issues/30359)**  
  
1. `undefined symbol: _ZN6tflite12tensor_utils39NeonMatrixBatchVectorMultiplyAccumulateEPKaiiS2_PKfiPfi`  
2. `Bus Error`
3. `ImportError: cannot import name 'cloud' from 'tensorflow.contrib' `
4. `export SetNumThreads to TFLite Python API`
5. `TensorFlow C binding for Raspberry Pi`
  
**Python API packages**

|Device|OS|Distribution|Architecture|Python ver|Note|
|:--|:--|:--|:--|:--|:--|
|RaspberryPi3/4|Raspbian/Debian|Stretch|armhf / armv7l|3.5.3|32bit|
|RaspberryPi3/4|Raspbian/Debian|Buster|armhf / armv7l|3.7.3|32bit|
|RaspberryPi3/4|Debian|Buster|aarch64 / armv8|3.7.3|64bit|

Bazel's pre-build binay is below.  
**https://github.com/PINTO0309/Bazel_bin.git**  

Cross compilation recommends using **`lhelontra`** repository.  
**https://github.com/lhelontra/tensorflow-on-arm.git**  

Prebuilt binary for Jetson Nano by **`Michael`**.  
**https://dl.photoprism.org/tensorflow/**

## Binary type
  
**Python 3.x + Tensorflow v1.14.0**  

|.whl|4Threads|Note|
|:--|:--:|:--|
|tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl|○|Raspbian/Debian Stretch|
|tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl|○|Raspbian/Debian Buster|
|tensorflow-1.14.0-cp37-cp37m-linux_aarch64.whl|○|Debian Buster|

**Python 3.x + Tensorflow v2**  

|.whl|4Threads|Note|
|:--|:--:|:--|
|tensorflow-2.0.0b1-cp35-cp35m-linux_armv7l.whl|○|Beta version1,Raspbian/Debian Stretch|
|tensorflow-2.0.0b1-cp37-cp37m-linux_armv7l.whl|○|Beta version1,Raspbian/Debian Buster|
|tensorflow-2.0.0b1-cp37-cp37m-linux_aarch64.whl|○|Beta version1,Debian Buster|

**【Appendix】 C Library + Tensorflow v1.x.x / v2.x.x**  
The behavior is unconfirmed because I do not have C language implementation skills.
```sh
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/C-library/1.14.0-armv7l/libtensorflow.tar.gz
$ tar -C /usr/local -xzf libtensorflow.tar.gz
$ rm libtensorflow.tar.gz
$ sudo ldconfig
```
|Version|Binary|Note|
|:--:|:--|:--|
|v1.14.0|C-library/1.14.0-armv7l/libtensorflow.tar.gz|Raspbian/Debian Buster armhf, glibc 2.28|
|v1.14.0|C-library/1.14.0-aarch64/libtensorflow.tar.gz|Debian Buster aarch64, glibc 2.28|
|v2.0.0-beta1|C-library/2.0.0beta1-armv7l/libtensorflow.tar.gz|Raspbian/Debian Buster, glibc 2.28|

## Usage
**Example of Python 3.x + Tensorflow v1 series**
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo apt-get install -y libatlas-base-dev
$ pip3 install -U --user six wheel mock
$ sudo pip3 uninstall tensorflow
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl
$ sudo pip3 install tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl

【Required】 Restart the terminal.
```
**Example of Python 3.x + Tensorflow v2 series**
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo apt-get install -y libatlas-base-dev
$ pip3 install -U --user six wheel mock
$ sudo apt update;sudo apt upgrade
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/tensorflow-2.0.0b1-cp35-cp35m-linux_armv7l.whl
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.0.0b1-cp35-cp35m-linux_armv7l.whl

【Required】 Restart the terminal.
```

## Operation check
**Example of Python 3.x series**
```bash
$ python3
>>> import tensorflow
>>> tensorflow.__version__
1.14.0
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
  
<details><summary>Tensorflow v1.11.0</summary><div>

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
  
</div></details>
  
<details><summary>Tensorflow v1.12.0</summary><div>
  
============================================================  
  
**Tensorflow v1.12.0 - Bazel 0.18.1**  

============================================================  
  
**Python3.x (Nov 15, 2018 Under construction)**  
  
- tensorflow/BUILD
```config
config_setting(
    name = "no_aws_support",
    define_values = {"no_aws_support": "false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_gcp_support",
    define_values = {"no_gcp_support": "false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_hdfs_support",
    define_values = {"no_hdfs_support": "false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_ignite_support",
    define_values = {"no_ignite_support": "false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_kafka_support",
    define_values = {"no_kafka_support": "false"},
    visibility = ["//visibility:public"],
)
```
- bazel.rc
```rc
# Options to disable default on features
build:noaws --define=no_aws_support=true
build:nogcp --define=no_gcp_support=true
build:nohdfs --define=no_hdfs_support=true
build:nokafka --define=no_kafka_support=true
build:noignite --define=no_ignite_support=true
```
- configure.py
```python:configure.py
  #set_build_var(environ_cp, 'TF_NEED_IGNITE', 'Apache Ignite',
  #              'with_ignite_support', True, 'ignite')


  ## On Windows, we don't have MKL support and the build is always monolithic.
  ## So no need to print the following message.
  ## TODO(pcloudy): remove the following if check when they make sense on Windows
  #if not is_windows():
  #  print('Preconfigured Bazel build configs. You can use any of the below by '
  #        'adding "--config=<>" to your build command. See tools/bazel.rc for '
  #        'more details.')
  #  config_info_line('mkl', 'Build with MKL support.')
  #  config_info_line('monolithic', 'Config for mostly static monolithic build.')
  #  config_info_line('gdr', 'Build with GDR support.')
  #  config_info_line('verbs', 'Build with libverbs support.')
  #  config_info_line('ngraph', 'Build with Intel nGraph support.')
  print('Preconfigured Bazel build configs. You can use any of the below by '
        'adding "--config=<>" to your build command. See .bazelrc for more '
        'details.')
  config_info_line('mkl', 'Build with MKL support.')
  config_info_line('monolithic', 'Config for mostly static monolithic build.')
  config_info_line('gdr', 'Build with GDR support.')
  config_info_line('verbs', 'Build with libverbs support.')
  config_info_line('ngraph', 'Build with Intel nGraph support.')

  print('Preconfigured Bazel build configs to DISABLE default on features:')
  config_info_line('noaws', 'Disable AWS S3 filesystem support.')
  config_info_line('nogcp', 'Disable GCP support.')
  config_info_line('nohdfs', 'Disable HDFS support.')
  config_info_line('noignite', 'Disable Apacha Ignite support.')
  config_info_line('nokafka', 'Disable Apache Kafka support.')
```

```tensorflow/contrib/BUILD
# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//third_party/mpi:mpi.bzl", "if_mpi")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow:tensorflow.bzl", "if_not_windows_cuda")

py_library(
    name = "contrib_py",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "**/*_test.py",
        ],
    ),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/all_reduce",
        "//tensorflow/contrib/batching:batch_py",
        "//tensorflow/contrib/bayesflow:bayesflow_py",
        "//tensorflow/contrib/boosted_trees:init_py",
        "//tensorflow/contrib/checkpoint/python:checkpoint",
        "//tensorflow/contrib/cluster_resolver:cluster_resolver_py",
        "//tensorflow/contrib/coder:coder_py",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/contrib/autograph",
        "//tensorflow/contrib/constrained_optimization",
        "//tensorflow/contrib/copy_graph:copy_graph_py",
        "//tensorflow/contrib/crf:crf_py",
        "//tensorflow/contrib/cudnn_rnn:cudnn_rnn_py",
        "//tensorflow/contrib/data",
        "//tensorflow/contrib/deprecated:deprecated_py",
        "//tensorflow/contrib/distribute:distribute",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow/contrib/estimator:estimator_py",
        "//tensorflow/contrib/factorization:factorization_py",
        "//tensorflow/contrib/feature_column:feature_column_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/gan",
        "//tensorflow/contrib/graph_editor:graph_editor_py",
        "//tensorflow/contrib/grid_rnn:grid_rnn_py",
        "//tensorflow/contrib/hadoop",
        "//tensorflow/contrib/hooks",
        "//tensorflow/contrib/image:distort_image_py",
        "//tensorflow/contrib/image:image_py",
        "//tensorflow/contrib/image:single_image_random_dot_stereograms_py",
        "//tensorflow/contrib/input_pipeline:input_pipeline_py",
        "//tensorflow/contrib/integrate:integrate_py",
        "//tensorflow/contrib/keras",
        "//tensorflow/contrib/kernel_methods",
        "//tensorflow/contrib/labeled_tensor",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/contrib/legacy_seq2seq:seq2seq_py",
        "//tensorflow/contrib/libsvm",
        "//tensorflow/contrib/linear_optimizer:sdca_estimator_py",
        "//tensorflow/contrib/linear_optimizer:sdca_ops_py",
        "//tensorflow/contrib/lite/python:lite",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/losses:metric_learning_py",
        "//tensorflow/contrib/memory_stats:memory_stats_py",
        "//tensorflow/contrib/meta_graph_transform",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/mixed_precision:mixed_precision",
        "//tensorflow/contrib/model_pruning",
        "//tensorflow/contrib/nccl:nccl_py",
        "//tensorflow/contrib/nearest_neighbor:nearest_neighbor_py",
        "//tensorflow/contrib/nn:nn_py",
        "//tensorflow/contrib/opt:opt_py",
        "//tensorflow/contrib/optimizer_v2:optimizer_v2_py",
        "//tensorflow/contrib/periodic_resample:init_py",
        "//tensorflow/contrib/predictor",
        "//tensorflow/contrib/proto",
        "//tensorflow/contrib/quantization:quantization_py",
        "//tensorflow/contrib/quantize:quantize_graph",
        "//tensorflow/contrib/receptive_field:receptive_field_py",
        "//tensorflow/contrib/recurrent:recurrent_py",
        "//tensorflow/contrib/reduce_slice_ops:reduce_slice_ops_py",
        "//tensorflow/contrib/remote_fused_graph/pylib:remote_fused_graph_ops_py",
        "//tensorflow/contrib/resampler:resampler_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/contrib/rpc",
        "//tensorflow/contrib/saved_model:saved_model_py",
        "//tensorflow/contrib/seq2seq:seq2seq_py",
        "//tensorflow/contrib/signal:signal_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/contrib/slim:nets",
        "//tensorflow/contrib/solvers:solvers_py",
        "//tensorflow/contrib/sparsemax:sparsemax_py",
        "//tensorflow/contrib/specs",
        "//tensorflow/contrib/staging",
        "//tensorflow/contrib/stat_summarizer:stat_summarizer_py",
        "//tensorflow/contrib/stateless",
        "//tensorflow/contrib/summary:summary",
        "//tensorflow/contrib/tensor_forest:init_py",
        "//tensorflow/contrib/tensorboard",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/contrib/text:text_py",
        "//tensorflow/contrib/tfprof",
        "//tensorflow/contrib/timeseries",
        "//tensorflow/contrib/tpu",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator:estimator_py",
    ] + if_mpi(["//tensorflow/contrib/mpi_collectives:mpi_collectives_py"]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_kafka_support": [],
        "//conditions:default": [
            "//tensorflow/contrib/kafka",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
             "//tensorflow/contrib/kinesis",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "//tensorflow/contrib/fused_conv:fused_conv_py",
             "//tensorflow/contrib/tensorrt:init_py",
             "//tensorflow/contrib/ffmpeg:ffmpeg_ops_py",
         ],
     }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "//tensorflow/contrib/bigtable",
            "//tensorflow/contrib/cloud:cloud_py",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_ignite_support": [],
        "//conditions:default": [
             "//tensorflow/contrib/ignite",
         ],
     }),
 )

cc_library(
    name = "contrib_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/boosted_trees:boosted_trees_kernels",
        "//tensorflow/contrib/coder:all_kernels",
        "//tensorflow/contrib/factorization/kernels:all_kernels",
        "//tensorflow/contrib/hadoop:dataset_kernels",
        "//tensorflow/contrib/input_pipeline:input_pipeline_ops_kernels",
        "//tensorflow/contrib/layers:sparse_feature_cross_op_kernel",
        "//tensorflow/contrib/nearest_neighbor:nearest_neighbor_ops_kernels",
        "//tensorflow/contrib/rnn:all_kernels",
        "//tensorflow/contrib/seq2seq:beam_search_ops_kernels",
        "//tensorflow/contrib/tensor_forest:model_ops_kernels",
        "//tensorflow/contrib/tensor_forest:stats_ops_kernels",
        "//tensorflow/contrib/tensor_forest:tensor_forest_kernels",
        "//tensorflow/contrib/text:all_kernels",
     ] + if_mpi(["//tensorflow/contrib/mpi_collectives:mpi_collectives_py"]) + if_cuda([
         "//tensorflow/contrib/nccl:nccl_kernels",
     ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
         "//tensorflow:linux_s390x": [],
         "//tensorflow:windows": [],
        "//tensorflow:no_kafka_support": [],
         "//conditions:default": [
             "//tensorflow/contrib/kafka:dataset_kernels",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
             "//tensorflow/contrib/kinesis:dataset_kernels",
         ],
    }) + if_not_windows([
        "//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
    ]),
 )

cc_library(
    name = "contrib_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/boosted_trees:boosted_trees_ops_op_lib",
        "//tensorflow/contrib/coder:all_ops",
        "//tensorflow/contrib/factorization:all_ops",
        "//tensorflow/contrib/framework:all_ops",
        "//tensorflow/contrib/hadoop:dataset_ops_op_lib",
        "//tensorflow/contrib/input_pipeline:input_pipeline_ops_op_lib",
        "//tensorflow/contrib/layers:sparse_feature_cross_op_op_lib",
        "//tensorflow/contrib/nccl:nccl_ops_op_lib",
        "//tensorflow/contrib/nearest_neighbor:nearest_neighbor_ops_op_lib",
        "//tensorflow/contrib/rnn:all_ops",
        "//tensorflow/contrib/seq2seq:beam_search_ops_op_lib",
        "//tensorflow/contrib/tensor_forest:model_ops_op_lib",
        "//tensorflow/contrib/tensor_forest:stats_ops_op_lib",
        "//tensorflow/contrib/tensor_forest:tensor_forest_ops_op_lib",
        "//tensorflow/contrib/text:all_ops",
        "//tensorflow/contrib/tpu:all_ops",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
         "//tensorflow:linux_s390x": [],
         "//tensorflow:windows": [],
        "//tensorflow:no_kafka_support": [],
         "//conditions:default": [
             "//tensorflow/contrib/kafka:dataset_ops_op_lib",
         ],
     }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "//tensorflow/contrib/kinesis:dataset_ops_op_lib",
        ],
    }) + if_not_windows([
        "//tensorflow/contrib/tensorrt:trt_engine_op_op_lib",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_ignite_support": [],
        "//conditions:default": [
             "//tensorflow/contrib/ignite:dataset_ops_op_lib",
         ],
     }),
 )
```
- tensorflow/core/platform/default/build_config.bzl
```
# Platform-specific build configurations.

load("@protobuf_archive//:protobuf.bzl", "proto_gen")
load("//tensorflow:tensorflow.bzl", "if_not_mobile")
load("//tensorflow:tensorflow.bzl", "if_windows")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)

# Appends a suffix to a list of deps.
def tf_deps(deps, suffix):
    tf_deps = []

    # If the package name is in shorthand form (ie: does not contain a ':'),
    # expand it to the full name.
    for dep in deps:
        tf_dep = dep

        if not ":" in dep:
            dep_pieces = dep.split("/")
            tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

        tf_deps += [tf_dep + suffix]

    return tf_deps

# Modified from @cython//:Tools/rules.bzl
def pyx_library(
        name,
        deps = [],
        py_deps = [],
        srcs = [],
        **kwargs):
    """Compiles a group of .pyx / .pxd / .py files.

    First runs Cython to create .cpp files for each input .pyx or .py + .pxd
    pair. Then builds a shared object for each, passing "deps" to each cc_binary
    rule (includes Python headers by default). Finally, creates a py_library rule
    with the shared objects and any pure Python "srcs", with py_deps as its
    dependencies; the shared objects can be imported like normal Python files.

    Args:
      name: Name for the rule.
      deps: C/C++ dependencies of the Cython (e.g. Numpy headers).
      py_deps: Pure Python dependencies of the final library.
      srcs: .py, .pyx, or .pxd files to either compile or pass through.
      **kwargs: Extra keyword arguments passed to the py_library.
    """

    # First filter out files that should be run compiled vs. passed through.
    py_srcs = []
    pyx_srcs = []
    pxd_srcs = []
    for src in srcs:
        if src.endswith(".pyx") or (src.endswith(".py") and
                                    src[:-3] + ".pxd" in srcs):
            pyx_srcs.append(src)
        elif src.endswith(".py"):
            py_srcs.append(src)
        else:
            pxd_srcs.append(src)
        if src.endswith("__init__.py"):
            pxd_srcs.append(src)

    # Invoke cython to produce the shared object libraries.
    for filename in pyx_srcs:
        native.genrule(
            name = filename + "_cython_translation",
            srcs = [filename],
            outs = [filename.split(".")[0] + ".cpp"],
            # Optionally use PYTHON_BIN_PATH on Linux platforms so that python 3
            # works. Windows has issues with cython_binary so skip PYTHON_BIN_PATH.
            cmd = "PYTHONHASHSEED=0 $(location @cython//:cython_binary) --cplus $(SRCS) --output-file $(OUTS)",
            tools = ["@cython//:cython_binary"] + pxd_srcs,
        )

    shared_objects = []
    for src in pyx_srcs:
        stem = src.split(".")[0]
        shared_object_name = stem + ".so"
        native.cc_binary(
            name = shared_object_name,
            srcs = [stem + ".cpp"],
            deps = deps + ["//third_party/python_runtime:headers"],
            linkshared = 1,
        )
        shared_objects.append(shared_object_name)

    # Now create a py_library with these shared objects as data.
    native.py_library(
        name = name,
        srcs = py_srcs,
        deps = py_deps,
        srcs_version = "PY2AND3",
        data = shared_objects,
        **kwargs
    )

def _proto_cc_hdrs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
    return ret

def _proto_cc_srcs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
    return ret

def _proto_py_outs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + "_pb2.py" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + "_pb2_grpc.py" for s in srcs]
    return ret

# Re-defined protocol buffer rule to allow building "header only" protocol
# buffers, to avoid duplicate registrations. Also allows non-iterable cc_libs
# containing select() statements.
def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        include = None,
        protoc = "@protobuf_archive//:protoc",
        internal_bootstrap_hack = False,
        use_grpc_plugin = False,
        use_grpc_namespace = False,
        default_header = False,
        **kargs):
    """Bazel rule to create a C++ protobuf library from proto source files.

    Args:
      name: the name of the cc_proto_library.
      srcs: the .proto files of the cc_proto_library.
      deps: a list of dependency labels; must be cc_proto_library.
      cc_libs: a list of other cc_library targets depended by the generated
          cc_library.
      include: a string indicating the include path of the .proto files.
      protoc: the label of the protocol compiler to generate the sources.
      internal_bootstrap_hack: a flag indicate the cc_proto_library is used only
          for bootstraping. When it is set to True, no files will be generated.
          The rule will simply be a provider for .proto files, so that other
          cc_proto_library can depend on it.
      use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
          when processing the proto files.
      default_header: Controls the naming of generated rules. If True, the `name`
          rule will be header-only, and an _impl rule will contain the
          implementation. Otherwise the header-only rule (name + "_headers_only")
          must be referred to explicitly.
      **kargs: other keyword arguments that are passed to cc_library.
    """

    includes = []
    if include != None:
        includes = [include]

    if internal_bootstrap_hack:
        # For pre-checked-in generated files, we add the internal_bootstrap_hack
        # which will skip the codegen action.
        proto_gen(
            name = name + "_genproto",
            srcs = srcs,
            includes = includes,
            protoc = protoc,
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in deps],
        )

        # An empty cc_library to make rule dependency consistent.
        native.cc_library(
            name = name,
            **kargs
        )
        return

    grpc_cpp_plugin = None
    plugin_options = []
    if use_grpc_plugin:
        grpc_cpp_plugin = "//external:grpc_cpp_plugin"
        if use_grpc_namespace:
            plugin_options = ["services_namespace=grpc"]

    gen_srcs = _proto_cc_srcs(srcs, use_grpc_plugin)
    gen_hdrs = _proto_cc_hdrs(srcs, use_grpc_plugin)
    outs = gen_srcs + gen_hdrs

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_cc = 1,
        includes = includes,
        plugin = grpc_cpp_plugin,
        plugin_language = "grpc",
        plugin_options = plugin_options,
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = [s + "_genproto" for s in deps],
    )

    if use_grpc_plugin:
        cc_libs += select({
            "//tensorflow:linux_s390x": ["//external:grpc_lib_unsecure"],
            "//conditions:default": ["//external:grpc_lib"],
        })

    if default_header:
        header_only_name = name
        impl_name = name + "_impl"
    else:
        header_only_name = name + "_headers_only"
        impl_name = name

    native.cc_library(
        name = impl_name,
        srcs = gen_srcs,
        hdrs = gen_hdrs,
        deps = cc_libs + deps,
        includes = includes,
        **kargs
    )
    native.cc_library(
        name = header_only_name,
        deps = ["@protobuf_archive//:protobuf_headers"] + if_static([impl_name]),
        hdrs = gen_hdrs,
        **kargs
    )

# Re-defined protocol buffer rule to bring in the change introduced in commit
# https://github.com/google/protobuf/commit/294b5758c373cbab4b72f35f4cb62dc1d8332b68
# which was not part of a stable protobuf release in 04/2018.
# TODO(jsimsa): Remove this once the protobuf dependency version is updated
# to include the above commit.
def py_proto_library(
        name,
        srcs = [],
        deps = [],
        py_libs = [],
        py_extra_srcs = [],
        include = None,
        default_runtime = "@protobuf_archive//:protobuf_python",
        protoc = "@protobuf_archive//:protoc",
        use_grpc_plugin = False,
        **kargs):
    """Bazel rule to create a Python protobuf library from proto source files

    NOTE: the rule is only an internal workaround to generate protos. The
    interface may change and the rule may be removed when bazel has introduced
    the native rule.

    Args:
      name: the name of the py_proto_library.
      srcs: the .proto files of the py_proto_library.
      deps: a list of dependency labels; must be py_proto_library.
      py_libs: a list of other py_library targets depended by the generated
          py_library.
      py_extra_srcs: extra source files that will be added to the output
          py_library. This attribute is used for internal bootstrapping.
      include: a string indicating the include path of the .proto files.
      default_runtime: the implicitly default runtime which will be depended on by
          the generated py_library target.
      protoc: the label of the protocol compiler to generate the sources.
      use_grpc_plugin: a flag to indicate whether to call the Python C++ plugin
          when processing the proto files.
      **kargs: other keyword arguments that are passed to cc_library.
    """
    outs = _proto_py_outs(srcs, use_grpc_plugin)

    includes = []
    if include != None:
        includes = [include]

    grpc_python_plugin = None
    if use_grpc_plugin:
        grpc_python_plugin = "//external:grpc_python_plugin"
        # Note: Generated grpc code depends on Python grpc module. This dependency
        # is not explicitly listed in py_libs. Instead, host system is assumed to
        # have grpc installed.

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_py = 1,
        includes = includes,
        plugin = grpc_python_plugin,
        plugin_language = "grpc",
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = [s + "_genproto" for s in deps],
    )

    if default_runtime and not default_runtime in py_libs + deps:
        py_libs = py_libs + [default_runtime]

    native.py_library(
        name = name,
        srcs = outs + py_extra_srcs,
        deps = py_libs + deps,
        imports = includes,
        **kargs
    )

def tf_proto_library_cc(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = [],
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_grpc_version = None,
        j2objc_api_version = 1,
        cc_api_version = 2,
        dart_api_version = 2,
        java_api_version = 2,
        py_api_version = 2,
        js_api_version = 2,
        js_codegen = "jspb",
        default_header = False):
    js_codegen = js_codegen  # unused argument
    js_api_version = js_api_version  # unused argument
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
        testonly = testonly,
        visibility = visibility,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    cc_deps = tf_deps(protodeps, "_cc")
    cc_name = name + "_cc"
    if not srcs:
        # This is a collection of sub-libraries. Build header-only and impl
        # libraries containing all the sources.
        proto_gen(
            name = cc_name + "_genproto",
            protoc = "@protobuf_archive//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in cc_deps],
        )
        native.cc_library(
            name = cc_name,
            deps = cc_deps + ["@protobuf_archive//:protobuf_headers"] + if_static([name + "_cc_impl"]),
            testonly = testonly,
            visibility = visibility,
        )
        native.cc_library(
            name = cc_name + "_impl",
            deps = [s + "_impl" for s in cc_deps] + ["@protobuf_archive//:cc_wkt_protos"],
        )

        return

    cc_proto_library(
        name = cc_name,
        testonly = testonly,
        srcs = srcs,
        cc_libs = cc_libs + if_static(
            ["@protobuf_archive//:protobuf"],
            ["@protobuf_archive//:protobuf_headers"],
        ),
        copts = if_not_windows([
            "-Wno-unknown-warning-option",
            "-Wno-unused-but-set-variable",
            "-Wno-sign-compare",
        ]),
        default_header = default_header,
        protoc = "@protobuf_archive//:protoc",
        use_grpc_plugin = use_grpc_plugin,
        visibility = visibility,
        deps = cc_deps + ["@protobuf_archive//:cc_wkt_protos"],
    )

def tf_proto_library_py(
        name,
        srcs = [],
        protodeps = [],
        deps = [],
        visibility = [],
        testonly = 0,
        srcs_version = "PY2AND3",
        use_grpc_plugin = False):
    py_deps = tf_deps(protodeps, "_py")
    py_name = name + "_py"
    if not srcs:
        # This is a collection of sub-libraries. Build header-only and impl
        # libraries containing all the sources.
        proto_gen(
            name = py_name + "_genproto",
            protoc = "@protobuf_archive//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in py_deps],
        )
        native.py_library(
            name = py_name,
            deps = py_deps + ["@protobuf_archive//:protobuf_python"],
            testonly = testonly,
            visibility = visibility,
        )
        return

    py_proto_library(
        name = py_name,
        testonly = testonly,
        srcs = srcs,
        default_runtime = "@protobuf_archive//:protobuf_python",
        protoc = "@protobuf_archive//:protoc",
        srcs_version = srcs_version,
        use_grpc_plugin = use_grpc_plugin,
        visibility = visibility,
        deps = deps + py_deps + ["@protobuf_archive//:protobuf_python"],
    )

def tf_jspb_proto_library(**kwargs):
    pass

def tf_nano_proto_library(**kwargs):
    pass

def tf_proto_library(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = [],
        testonly = 0,
        cc_libs = [],
        cc_api_version = 2,
        cc_grpc_version = None,
        dart_api_version = 2,
        j2objc_api_version = 1,
        java_api_version = 2,
        py_api_version = 2,
        js_api_version = 2,
        js_codegen = "jspb",
        provide_cc_alias = False,
        default_header = False):
    """Make a proto library, possibly depending on other proto libraries."""
    _ignore = (js_api_version, js_codegen, provide_cc_alias)

    tf_proto_library_cc(
        name = name,
        testonly = testonly,
        srcs = srcs,
        cc_grpc_version = cc_grpc_version,
        cc_libs = cc_libs,
        default_header = default_header,
        protodeps = protodeps,
        visibility = visibility,
    )

    tf_proto_library_py(
        name = name,
        testonly = testonly,
        srcs = srcs,
        protodeps = protodeps,
        srcs_version = "PY2AND3",
        use_grpc_plugin = has_services,
        visibility = visibility,
    )

# A list of all files under platform matching the pattern in 'files'. In
# contrast with 'tf_platform_srcs' below, which seletive collects files that
# must be compiled in the 'default' platform, this is a list of all headers
# mentioned in the platform/* files.
def tf_platform_hdrs(files):
    return native.glob(["platform/*/" + f for f in files])

def tf_platform_srcs(files):
    base_set = ["platform/default/" + f for f in files]
    windows_set = base_set + ["platform/windows/" + f for f in files]
    posix_set = base_set + ["platform/posix/" + f for f in files]

    # Handle cases where we must also bring the posix file in. Usually, the list
    # of files to build on windows builds is just all the stuff in the
    # windows_set. However, in some cases the implementations in 'posix/' are
    # just what is necessary and historically we choose to simply use the posix
    # file instead of making a copy in 'windows'.
    for f in files:
        if f == "error.cc":
            windows_set.append("platform/posix/" + f)

    return select({
        "//tensorflow:windows": native.glob(windows_set),
        "//conditions:default": native.glob(posix_set),
    })

def tf_additional_lib_hdrs(exclude = []):
    windows_hdrs = native.glob([
        "platform/default/*.h",
        "platform/windows/*.h",
        "platform/posix/error.h",
    ], exclude = exclude)
    return select({
        "//tensorflow:windows": windows_hdrs,
        "//conditions:default": native.glob([
            "platform/default/*.h",
            "platform/posix/*.h",
        ], exclude = exclude),
    })

def tf_additional_lib_srcs(exclude = []):
    windows_srcs = native.glob([
        "platform/default/*.cc",
        "platform/windows/*.cc",
        "platform/posix/error.cc",
    ], exclude = exclude)
    return select({
        "//tensorflow:windows": windows_srcs,
        "//conditions:default": native.glob([
            "platform/default/*.cc",
            "platform/posix/*.cc",
        ], exclude = exclude),
    })

def tf_additional_minimal_lib_srcs():
    return [
        "platform/default/integral_types.h",
        "platform/default/mutex.h",
    ]

def tf_additional_proto_hdrs():
    return [
        "platform/default/integral_types.h",
        "platform/default/logging.h",
        "platform/default/protobuf.h",
    ] + if_windows([
        "platform/windows/integral_types.h",
    ])

def tf_additional_proto_compiler_hdrs():
    return [
        "platform/default/protobuf_compiler.h",
    ]

def tf_additional_proto_srcs():
    return [
        "platform/default/protobuf.cc",
    ]

def tf_additional_human_readable_json_deps():
    return []

def tf_additional_all_protos():
    return ["//tensorflow/core:protos_all"]

def tf_protos_all_impl():
    return ["//tensorflow/core:protos_all_cc_impl"]

def tf_protos_all():
    return if_static(
        extra_deps = tf_protos_all_impl(),
        otherwise = ["//tensorflow/core:protos_all_cc"],
    )

def tf_protos_grappler_impl():
    return ["//tensorflow/core/grappler/costs:op_performance_data_cc_impl"]

def tf_protos_grappler():
    return if_static(
        extra_deps = tf_protos_grappler_impl(),
        otherwise = ["//tensorflow/core/grappler/costs:op_performance_data_cc"],
    )

def tf_additional_cupti_wrapper_deps():
    return ["//tensorflow/core/platform/default/gpu:cupti_wrapper"]

def tf_additional_device_tracer_srcs():
    return ["platform/default/device_tracer.cc"]

def tf_additional_device_tracer_cuda_deps():
    return []

def tf_additional_device_tracer_deps():
    return []

def tf_additional_libdevice_data():
    return []

def tf_additional_libdevice_deps():
    return ["@local_config_cuda//cuda:cuda_headers"]

def tf_additional_libdevice_srcs():
    return ["platform/default/cuda_libdevice_path.cc"]

def tf_additional_test_deps():
    return []

def tf_additional_test_srcs():
    return [
        "platform/default/test_benchmark.cc",
    ] + select({
        "//tensorflow:windows": [
            "platform/windows/test.cc",
        ],
        "//conditions:default": [
            "platform/posix/test.cc",
        ],
    })

def tf_kernel_tests_linkstatic():
    return 0

def tf_additional_lib_defines():
    """Additional defines needed to build TF libraries."""
    return []

def tf_additional_lib_deps():
    """Additional dependencies needed to build TF libraries."""
    return [
        "@com_google_absl//absl/base:base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:optional",
    ] + if_static(
        ["@nsync//:nsync_cpp"],
        ["@nsync//:nsync_headers"],
    )

def tf_additional_core_deps():
     return select({
         "//tensorflow:android": [],
         "//tensorflow:ios": [],
         "//tensorflow:linux_s390x": [],
         "//tensorflow:windows": [],
         "//tensorflow:no_gcp_support": [],
         "//conditions:default": [
             "//tensorflow/core/platform/cloud:gcs_file_system",
         ],
     }) + select({
         "//tensorflow:android": [],
         "//tensorflow:ios": [],
         "//tensorflow:linux_s390x": [],
         "//tensorflow:windows": [],
         "//tensorflow:no_hdfs_support": [],
         "//conditions:default": [
             "//tensorflow/core/platform/hadoop:hadoop_file_system",
         ],
     }) + select({
         "//tensorflow:android": [],
         "//tensorflow:ios": [],
         "//tensorflow:linux_s390x": [],
         "//tensorflow:windows": [],
         "//tensorflow:no_aws_support": [],
         "//conditions:default": [
             "//tensorflow/core/platform/s3:s3_file_system",
         ],
     })
 
# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_op_deps():
    return select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
           "//tensorflow/contrib/cloud:bigquery_reader_ops_op_lib",
           "//tensorflow/contrib/cloud:gcs_config_ops_op_lib",
       ],
   })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_kernel_deps():
    return select({
        "//tensorflow:android": [],
        "//tensorflow:windows": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//conditions:default": [
            "//tensorflow/contrib/cloud/kernels:bigquery_reader_ops",
            "//tensorflow/contrib/cloud/kernels:gcs_config_ops",
        ],
    })

def tf_lib_proto_parsing_deps():
    return [
        ":protos_all_cc",
        "//third_party/eigen3",
        "//tensorflow/core/platform/default/build_config:proto_parsing",
    ]

def tf_lib_proto_compiler_deps():
    return [
        "@protobuf_archive//:protoc_lib",
    ]

def tf_additional_verbs_lib_defines():
    return select({
        "//tensorflow:with_verbs_support": ["TENSORFLOW_USE_VERBS"],
        "//conditions:default": [],
    })

def tf_additional_mpi_lib_defines():
    return select({
        "//tensorflow:with_mpi_support": ["TENSORFLOW_USE_MPI"],
        "//conditions:default": [],
    })

def tf_additional_gdr_lib_defines():
    return select({
        "//tensorflow:with_gdr_support": ["TENSORFLOW_USE_GDR"],
        "//conditions:default": [],
    })

def tf_py_clif_cc(name, visibility = None, **kwargs):
    pass

def tf_pyclif_proto_library(
        name,
        proto_lib,
        proto_srcfile = "",
        visibility = None,
        **kwargs):
    pass

def tf_additional_binary_deps():
    return ["@nsync//:nsync_cpp"] + if_cuda(
        [
            "//tensorflow/stream_executor:cuda_platform",
            "//tensorflow/core/platform/default/build_config:cuda",
        ],
    ) + [
        # TODO(allenl): Split these out into their own shared objects (they are
        # here because they are shared between contrib/ op shared objects and
        # core).
        "//tensorflow/core/kernels:lookup_util",
        "//tensorflow/core/util/tensor_bundle",
    ] + if_mkl_ml(
        [
            "//third_party/mkl:intel_binary_blob",
        ],
    )
```
- tensorflow/tools/lib_package/BUILD
```
# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer files.

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow:tensorflow.bzl", "tf_binary_additional_srcs")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//third_party/mkl:build_defs.bzl", "if_mkl")

genrule(
    name = "libtensorflow_proto",
    srcs = ["//tensorflow/core:protos_all_proto_srcs"],
    outs = ["libtensorflow_proto.zip"],
    cmd = "zip $@ $(SRCS)",
)

pkg_tar(
    name = "libtensorflow",
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":cheaders",
        ":clib",
        ":clicenses",
        ":eager_cheaders",
    ],
)

pkg_tar(
    name = "libtensorflow_jni",
    extension = "tar.gz",
    files = [
        "include/tensorflow/jni/LICENSE",
        "//tensorflow/java:libtensorflow_jni",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

# Shared objects that all TensorFlow libraries depend on.
pkg_tar(
    name = "common_deps",
    files = tf_binary_additional_srcs(),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    files = [
        "//tensorflow/c:headers",
    ],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "eager_cheaders",
    files = [
        "//tensorflow/c/eager:headers",
    ],
    package_dir = "include/tensorflow/c/eager",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "clib",
    files = ["//tensorflow:libtensorflow.so"],
    package_dir = "lib",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

pkg_tar(
    name = "clicenses",
    files = [":include/tensorflow/c/LICENSE"],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@llvm//:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({

        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/nanopb:LICENSE.txt",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/c/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@icu//:icu4j/main/shared/licenses/LICENSE",
        "@jpeg//:LICENSE.md",
        "@llvm//:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]),
    outs = ["include/tensorflow/jni/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

sh_test(
    name = "libtensorflow_test",
    size = "small",
    srcs = ["libtensorflow_test.sh"],
    data = [
        "libtensorflow_test.c",
        ":libtensorflow.tar.gz",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)

sh_test(
    name = "libtensorflow_java_test",
    size = "small",
    srcs = ["libtensorflow_java_test.sh"],
    data = [
        ":LibTensorFlowTest.java",
        ":libtensorflow_jni.tar.gz",
        "//tensorflow/java:libtensorflow.jar",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)
```
- tensorflow/tools/pip_package/BUILD
```
# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_windows",
    "transitive_hdrs",
)
load("//third_party/mkl:build_defs.bzl", "if_mkl", "if_mkl_ml")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow/core:platform/default/build_config_root.bzl", "tf_additional_license_deps")
load(
    "//third_party/ngraph:build_defs.bzl",
    "if_ngraph",
)

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.
transitive_hdrs(
    name = "included_headers",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
        "//third_party/eigen3",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

COMMON_PIP_DEPS = [
    ":licenses",
    "MANIFEST.in",
    "README",
    "setup.py",
    ":included_headers",
    "//tensorflow:tensorflow_py",
    "//tensorflow/contrib/autograph:autograph",
    "//tensorflow/contrib/boosted_trees:boosted_trees_pip",
    "//tensorflow/contrib/cluster_resolver:cluster_resolver_pip",
    "//tensorflow/contrib/compiler:xla",
    "//tensorflow/contrib/constrained_optimization:constrained_optimization_pip",
    "//tensorflow/contrib/eager/python/examples:examples_pip",
    "//tensorflow/contrib/eager/python:evaluator",
    "//tensorflow/contrib/gan:gan",
    "//tensorflow/contrib/graph_editor:graph_editor_pip",
    "//tensorflow/contrib/keras:keras",
    "//tensorflow/contrib/labeled_tensor:labeled_tensor_pip",
    "//tensorflow/contrib/nn:nn_py",
    "//tensorflow/contrib/predictor:predictor_pip",
    "//tensorflow/contrib/proto:proto",
    "//tensorflow/contrib/receptive_field:receptive_field_pip",
    "//tensorflow/contrib/rate:rate",
    "//tensorflow/contrib/rpc:rpc_pip",
    "//tensorflow/contrib/session_bundle:session_bundle_pip",
    "//tensorflow/contrib/signal:signal_py",
    "//tensorflow/contrib/signal:test_util",
    "//tensorflow/contrib/slim:slim",
    "//tensorflow/contrib/slim/python/slim/data:data_pip",
    "//tensorflow/contrib/slim/python/slim/nets:nets_pip",
    "//tensorflow/contrib/specs:specs",
    "//tensorflow/contrib/summary:summary_test_util",
    "//tensorflow/contrib/tensor_forest:init_py",
    "//tensorflow/contrib/tensor_forest/hybrid:hybrid_pip",
    "//tensorflow/contrib/timeseries:timeseries_pip",
    "//tensorflow/contrib/tpu",
    "//tensorflow/examples/tutorials/mnist:package",
    # "//tensorflow/python/autograph/converters:converters",
    # "//tensorflow/python/autograph/core:core",
    "//tensorflow/python/autograph/core:test_lib",
    # "//tensorflow/python/autograph/impl:impl",
    # "//tensorflow/python/autograph/lang:lang",
    # "//tensorflow/python/autograph/operators:operators",
    # "//tensorflow/python/autograph/pyct:pyct",
    # "//tensorflow/python/autograph/pyct/testing:testing",
    # "//tensorflow/python/autograph/pyct/static_analysis:static_analysis",
    "//tensorflow/python/autograph/pyct/common_transformers:common_transformers",
    "//tensorflow/python:cond_v2",
    "//tensorflow/python:distributed_framework_test_lib",
    "//tensorflow/python:meta_graph_testdata",
    "//tensorflow/python:spectral_ops_test_util",
    "//tensorflow/python:util_example_parser_configuration",
    "//tensorflow/python/data/experimental/kernel_tests/serialization:dataset_serialization_test_base",
    "//tensorflow/python/data/experimental/kernel_tests:stats_dataset_test_base",
    "//tensorflow/python/data/kernel_tests:test_base",
    "//tensorflow/python/debug:debug_pip",
    "//tensorflow/python/eager:eager_pip",
    "//tensorflow/python/kernel_tests/testdata:self_adjoint_eig_op_test_files",
    "//tensorflow/python/saved_model:saved_model",
    "//tensorflow/python/tools:tools_pip",
    "//tensorflow/python/tools/api/generator:create_python_api",
    "//tensorflow/python:test_ops",
    "//tensorflow/python:while_v2",
    "//tensorflow/tools/dist_test/server:grpc_tensorflow_server",
]

# On Windows, python binary is a zip file of runfiles tree.
# Add everything to its data dependency for generating a runfiles tree
# for building the pip package on Windows.
py_binary(
    name = "simple_console_for_windows",
    srcs = ["simple_console_for_windows.py"],
    data = COMMON_PIP_DEPS,
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

filegroup(
    name = "licenses",
    data = [
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "//third_party/hadoop:LICENSE.txt",
        "@absl_py//absl/flags:LICENSE",
        "@arm_neon_2_x86_sse//:LICENSE",
        "@astor_archive//:LICENSE",
        "@boringssl//:LICENSE",
        "@com_google_absl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@flatbuffers//:LICENSE.txt",
        "@gast_archive//:PKG-INFO",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@pcre//:LICENCE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@six_archive//:LICENSE",
        "@snappy//:COPYING",
        "@swig//:LICENSE",
        "@termcolor_archive//:COPYING.txt",
        "@zlib_archive//:zlib.h",
        "@org_python_pypi_backports_weakref//:LICENSE",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googleapis_googleapis//:LICENSE",
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_kafka_support": [],
        "//conditions:default": [
            "@kafka//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/nanopb:LICENSE.txt",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ) + if_ngraph([
        "@ngraph//:LICENSE",
        "@ngraph_tf//:LICENSE",
        "@nlohmann_json_lib//:LICENSE.MIT",
        "@tbb//:LICENSE",
    ]) + tf_additional_license_deps(),
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = select({
        "//tensorflow:windows": [
            ":simple_console_for_windows",
            "//tensorflow/contrib/lite/python:interpreter_test_data",
            "//tensorflow/contrib/lite/python:tflite_convert",
            "//tensorflow/contrib/lite/toco/python:toco_from_protos",
        ],
        "//conditions:default": COMMON_PIP_DEPS + [
            ":simple_console",
            "//tensorflow/contrib/lite/python:interpreter_test_data",
            "//tensorflow/contrib/lite/python:tflite_convert",
            "//tensorflow/contrib/lite/toco/python:toco_from_protos",
        ],
    }) + if_mkl_ml(["//third_party/mkl:intel_binary_blob"]),
)

# A genrule for generating a marker file for the pip package on Windows
#
# This only works on Windows, because :simple_console_for_windows is a
# python zip file containing everything we need for building the pip package.
# However, on other platforms, due to https://github.com/bazelbuild/bazel/issues/4223,
# when C++ extensions change, this generule doesn't rebuild.
genrule(
    name = "win_pip_package_marker",
    srcs = if_windows([
        ":build_pip_package",
        ":simple_console_for_windows",
    ]),
    outs = ["win_pip_package_marker_file"],
    cmd = select({
        "//conditions:default": "touch $@",
        "//tensorflow:windows": "md5sum $(locations :build_pip_package) $(locations :simple_console_for_windows) > $@",
    }),
    visibility = ["//visibility:public"],
)
```
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
WARNING: Processed legacy workspace file /home/pi/work/tensorflow/tools/bazel.rc. This file will not be processed in the next release of Bazel. Please read https://github.com/bazelbuild/bazel/issues/6319 for further information, including how to upgrade.
WARNING: Running Bazel server needs to be killed, because the startup options are different.
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.18.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /home/pi/inference_engine_vpu_arm/python/python3.5/armv7l
  /usr/local/lib
  /home/pi/inference_engine_vpu_arm/python/python3.5
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/home/pi/inference_engine_vpu_arm/python/python3.5/armv7l]
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

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apacha Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
Configuration finished
```
~~https://github.com/tensorflow/tensorflow/issues/22819~~  
~~https://github.com/tensorflow/tensorflow/commit/d80eb525e94763e09cbb9fa3cbef9a0f64e2cb2a~~  
~~https://github.com/tensorflow/tensorflow/commit/5847293aeb9ab45a02c4231c40569a15bd4541c6~~  
https://github.com/tensorflow/tensorflow/issues/23721  
https://github.com/tensorflow/tensorflow/pull/25748  
https://github.com/tensorflow/tensorflow/issues/25120#issuecomment-464296755  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package  
https://github.com/tensorflow/tensorflow/issues/24372  
https://gist.github.com/fyhertz/4cef0b696b37d38964801d3ef21e8ce2  
```
$ sudo bazel --host_jvm_args=-Xmx512m build \
--config=opt \
--config=noaws \
--config=nogcp \
--config=nohdfs \
--config=noignite \
--config=nokafka \
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
  
</div></details>
  
<details><summary>Tensorflow v1.13.1</summary><div>
  
============================================================  
  
**Tensorflow v1.13.1 - Bazel 0.19.2**  

============================================================  
  
**Python3.x**  
  

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
import sys
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.util.lazy_loader import LazyLoader
  from tensorflow.python.util.tf_export import tf_export as _tf_export

  # Lazy load since some of the performance benchmark skylark rules
  # break dependencies. Must use double quotes to match code internal rewrite
  # rule.
  # pylint: disable=g-inconsistent-quotes
  _interpreter_wrapper = LazyLoader(
      "_interpreter_wrapper", globals(),
      "tensorflow.lite.python.interpreter_wrapper."
      "tensorflow_wrap_interpreter_wrapper")
  # pylint: enable=g-inconsistent-quotes

  del LazyLoader
except ImportError:
  # When full Tensorflow Python PIP is not available do not use lazy load
  # and instead uf the tflite_runtime path.
  from tflite_runtime.lite.python import interpreter_wrapper as _interpreter_wrapper

  def tf_export_dummy(*x, **kwargs):
    del x, kwargs
    return lambda x: x
  _tf_export = tf_export_dummy


@_tf_export('lite.Interpreter')
class Interpreter(object):
  """Interpreter inferace for TF-Lite Models."""

  def __init__(self, model_path=None, model_content=None):
    """Constructor.
    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.
    Raises:
      ValueError: If the interpreter was unable to create.
    """
    if model_path and not model_content:
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromFile(
              model_path))
      if not self._interpreter:
        raise ValueError('Failed to open {}'.format(model_path))
    elif model_content and not model_path:
      # Take a reference, so the pointer remains valid.
      # Since python strings are immutable then PyString_XX functions
      # will always return the same pointer.
      self._model_content = model_content
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromBuffer(
              model_content))
    elif not model_path and not model_path:
      raise ValueError('`model_path` or `model_content` must be specified.')
    else:
      raise ValueError('Can\'t both provide `model_path` and `model_content`')

  def allocate_tensors(self):
    self._ensure_safe()
    return self._interpreter.AllocateTensors()

  def _safe_to_run(self):
    """Returns true if there exist no numpy array buffers.
    This means it is safe to run tflite calls that may destroy internally
    allocated memory. This works, because in the wrapper.cc we have made
    the numpy base be the self._interpreter.
    """
    # NOTE, our tensor() call in cpp will use _interpreter as a base pointer.
    # If this environment is the only _interpreter, then the ref count should be
    # 2 (1 in self and 1 in temporary of sys.getrefcount).
    return sys.getrefcount(self._interpreter) == 2

  def _ensure_safe(self):
    """Makes sure no numpy arrays pointing to internal buffers are active.
    This should be called from any function that will call a function on
    _interpreter that may reallocate memory e.g. invoke(), ...
    Raises:
      RuntimeError: If there exist numpy objects pointing to internal memory
        then we throw.
    """
    if not self._safe_to_run():
      raise RuntimeError("""There is at least 1 reference to internal data
      in the interpreter in the form of a numpy array or slice. Be sure to
      only hold the function returned from tensor() if you are using raw
      data access.""")

  def _get_tensor_details(self, tensor_index):
    """Gets tensor details.
    Args:
      tensor_index: Tensor index of tensor to query.
    Returns:
      a dictionary containing the name, index, shape and type of the tensor.
    Raises:
      ValueError: If tensor_index is invalid.
    """
    tensor_index = int(tensor_index)
    tensor_name = self._interpreter.TensorName(tensor_index)
    tensor_size = self._interpreter.TensorSize(tensor_index)
    tensor_type = self._interpreter.TensorType(tensor_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index)

    if not tensor_name or not tensor_type:
      raise ValueError('Could not get tensor details')

    details = {
        'name': tensor_name,
        'index': tensor_index,
        'shape': tensor_size,
        'dtype': tensor_type,
        'quantization': tensor_quantization,
    }

    return details

  def get_tensor_details(self):
    """Gets tensor details for every tensor with valid tensor details.
    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.
    Returns:
      A list of dictionaries containing tensor information.
    """
    tensor_details = []
    for idx in range(self._interpreter.NumTensors()):
      try:
        tensor_details.append(self._get_tensor_details(idx))
      except ValueError:
        pass
    return tensor_details

  def get_input_details(self):
    """Gets model input details.
    Returns:
      A list of input details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.InputIndices()
    ]

  def set_tensor(self, tensor_index, value):
    """Sets the value of the input tensor. Note this copies data in `value`.
    If you want to avoid copying, you can use the `tensor()` function to get a
    numpy buffer pointing to the input buffer in the tflite interpreter.
    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
                    the 'index' field in get_input_details.
      value: Value of tensor to set.
    Raises:
      ValueError: If the interpreter could not set the tensor.
    """
    self._interpreter.SetTensor(tensor_index, value)

  def resize_tensor_input(self, input_index, tensor_size):
    """Resizes an input tensor.
    Args:
      input_index: Tensor index of input to set. This value can be gotten from
                   the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.
    Raises:
      ValueError: If the interpreter could not resize the input tensor.
    """
    self._ensure_safe()
    # `ResizeInputTensor` now only accepts int32 numpy array as `tensor_size
    # parameter.
    tensor_size = np.array(tensor_size, dtype=np.int32)
    self._interpreter.ResizeInputTensor(input_index, tensor_size)

  def get_output_details(self):
    """Gets model output details.
    Returns:
      A list of output details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.OutputIndices()
    ]

  def get_tensor(self, tensor_index):
    """Gets the value of the input tensor (get a copy).
    If you wish to avoid the copy, use `tensor()`. This function cannot be used
    to read intermediate results.
    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.
    Returns:
      a numpy array.
    """
    return self._interpreter.GetTensor(tensor_index)

  def tensor(self, tensor_index):
    """Returns function that gives a numpy view of the current tensor buffer.
    This allows reading and writing to this tensors w/o copies. This more
    closely mirrors the C++ Interpreter class interface's tensor() member, hence
    the name. Be careful to not hold these output references through calls
    to `allocate_tensors()` and `invoke()`. This function cannot be used to read
    intermediate results.
    Usage:
    ```
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    for i in range(10):
      input().fill(3.)
      interpreter.invoke()
      print("inference %s" % output())
    ```
    Notice how this function avoids making a numpy array directly. This is
    because it is important to not hold actual numpy views to the data longer
    than necessary. If you do, then the interpreter can no longer be invoked,
    because it is possible the interpreter would resize and invalidate the
    referenced tensors. The NumPy API doesn't allow any mutability of the
    the underlying buffers.
    WRONG:
    ```
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])()
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    interpreter.allocate_tensors()  # This will throw RuntimeError
    for i in range(10):
      input.fill(3.)
      interpreter.invoke()  # this will throw RuntimeError since input,output
    ```
    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.
    Returns:
      A function that can return a new numpy array pointing to the internal
      TFLite tensor state at any point. It is safe to hold the function forever,
      but it is not safe to hold the numpy array forever.
    """
    return lambda: self._interpreter.tensor(self._interpreter, tensor_index)

  def invoke(self):
    """Invoke the interpreter.
    Be sure to set the input sizes, allocate tensors and fill values before
    calling this.
    Raises:
      ValueError: When the underlying interpreter fails raise ValueError.
    """
    self._ensure_safe()
    self._interpreter.Invoke()

  def reset_all_variables(self):
    return self._interpreter.ResetVariableTensors()

  def set_num_threads(self, i):
    """Set number of threads used by TFLite kernels.
    If not set, kernels are running single-threaded. Note that currently,
    only some kernels, such as conv, are multithreaded.
    Args:
      i: number of threads.
    """
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
```
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
    ],
)
```
- tensorflow/tensorflow/core/kernels/BUILD - Delete the following
```
tf_kernel_library(
    name = "matrix_square_root_op",
    prefix = "matrix_square_root_op",
    deps = LINALG_DEPS,
)
```
- tensorflow/lite/tools/make/Makefile
```
BUILD_WITH_NNAPI=false
ifeq ($(BUILD_WITH_NNAPI),true)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/nnapi_delegate_disabled.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/nnapi_delegate.cc
endif

ifeq ($(TARGET),ios)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_default.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_ios.cc
endif
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
$ sudo cp /tmp/tensorflow_pkg/tensorflow-1.13.1-cp35-cp35m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-1.13.1-cp35-cp35m-linux_armv7l.whl 
```
  
</div></details>
  
<details><summary>Tensorflow v1.14.0</summary><div>
  
============================================================  
  
**Tensorflow v1.14.0 - Bazel 0.24.1 - Stretch - armhf**  

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

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev openjdk-8-jdk

$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock
$ sudo apt update;sudo apt upgrade

$ cd ~
$ git clone https://github.com/PINTO0309/Bazel_bin.git
$ cd Bazel_bin
$ ./0.24.1/Raspbian_Stretch_armhf/install.sh

$ cd ~
$ git clone -b v1.14.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.14.0
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
You have bazel 0.24.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /home/pi/inference_engine_vpu_arm/python/python3.5
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
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
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
$ sudo cp /tmp/tensorflow_pkg/tensorflow-1.14.0-cp35-cp35m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-1.14.0-cp35-cp35m-linux_armv7l.whl 
```
  
============================================================  
  
**Tensorflow v1.14.0 - Bazel 0.24.1 - Buster - armhf**  

============================================================  
First, prepare an emulation environment for armhf with QEMU 4.0.0. (CPU 4core, RAM 4GB)  
**[How to create a Debian Buster armhf OS image from scratch in hardware emulation mode of QEMU 4.0.0 (Kernel 4.19.0-5-armmp-lpae, for building Tensorflow armhf)](https://qiita.com/PINTO/items/c10283a28d0699f01e01)**  
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev openjdk-8-jdk

$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/numpy-1.16.4-cp37-cp37m-linux_armv7l.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/h5py-2.9.0-cp37-cp37m-linux_armv7l.whl
$ sudo pip3 install numpy-1.16.4-cp37-cp37m-linux_armv7l.whl
$ sudo pip3 install h5py-2.9.0-cp37-cp37m-linux_armv7l.whl
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six wheel mock
$ sudo apt update;sudo apt upgrade

$ cd ~
$ git clone https://github.com/PINTO0309/Bazel_bin.git
$ cd Bazel_bin
$ ./0.24.1/Raspbian_Buster_armhf/install.sh

$ cd ~
$ git clone -b v1.14.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.14.0
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
- tensorflow/lite/tools/make/Makefile
```python
BUILD_WITH_NNAPI=false
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib import checkpoint
#if os.name != "nt" and platform.machine() != "s390x":
#  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib.summary import summary

if os.name != "nt" and platform.machine() != "s390x":
  try:
    from tensorflow.contrib import cloud
  except ImportError:
    pass

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
```
- configure
```bash
$ ./configure 
Extracting Bazel installation...
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.24.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /home/pi/inference_engine_vpu_arm/python/python3.5
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
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```
- build
```bash
$ sudo bazel build \
--config=opt \
--config=noaws \
--config=nogcp \
--config=nohdfs \
--config=noignite \
--config=nokafka \
--config=nonccl \
--local_resources=4096.0,2.0,1.0 \
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
$ sudo cp /tmp/tensorflow_pkg/tensorflow-1.14.0-cp37-cp37m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-1.14.0-cp37-cp37m-linux_armv7l.whl 
```
  
============================================================  
  
**Tensorflow v1.14.0 - Bazel 0.24.1 - Buster - aarch64**  

============================================================  
  
First, prepare an emulation environment for aarch64 with QEMU 4.0.0.  
**[How to create a Debian Buster aarch64 OS image from scratch in QEMU 4.0.0 hardware emulation mode (Kernel 4.19.0-5-arm64, for Tensorflow aarch64 build)](https://qiita.com/PINTO/items/e117bb0389f2163e2ac8)**
  
Next, build Bazel and Tensorflow according to the following procedure in the emulator environment.
```bash
$ sudo apt-get install -y \
libhdf5-dev libc-ares-dev libeigen3-dev \
libatlas3-base net-tools build-essential \
zip unzip python3-pip curl wget git zip unzip
$ sudo pip3 install pip --upgrade
$ sudo pip3 install zipper
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/absl_py-0.7.1-cp37-none-any.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/gast-0.2.2-cp37-none-any.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/grpcio-1.21.1-cp37-cp37m-linux_aarch64.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/h5py-2.9.0-cp37-cp37m-linux_aarch64.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/numpy-1.16.4-cp37-cp37m-linux_aarch64.whl
$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/packages/wrapt-1.11.2-cp37-cp37m-linux_aarch64.whl
$ sudo pip3 install *.whl
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo pip3 install -U --user mock zipper wheel

$ sudo apt-get update
$ sudo apt-get remove -y openjdk-8* --purge
$ sudo apt-get install -y openjdk-11-jdk

$ cd ~
$ mkdir bazel;cd bazel
$ wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip
$ unzip bazel-0.24.1-dist.zip
$ env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk"

$ nano compile.sh

#################################################################################
bazel_build "src:bazel_nojdk${EXE_EXT}" \
  --action_env=PATH \
  --host_platform=@bazel_tools//platforms:host_platform \
  --platforms=@bazel_tools//platforms:target_platform \
  || fail "Could not build Bazel"
#################################################################################
↓
#################################################################################
bazel_build "src:bazel_nojdk${EXE_EXT}" \
  --host_javabase=@local_jdk//:jdk \
  --action_env=PATH \
  --host_platform=@bazel_tools//platforms:host_platform \
  --platforms=@bazel_tools//platforms:target_platform \
  || fail "Could not build Bazel"
#################################################################################

$ sudo bash ./compile.sh
$ sudo cp output/bazel /usr/local/bin

$ bazel version
Extracting Bazel installation...
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
Build label: 0.24.1- (@non-git)
Build target: bazel-out/aarch64-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Sun Jun 23 20:46:48 2019 (1561322808)
Build timestamp: 1561322808
Build timestamp as int: 1561322808

$ cd ~
$ git clone -b v1.14.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.14.0
```
- tensorflow/lite/python/interpreter.py
```bash
# Add the following two lines to the last line
  def set_num_threads(self, i):
    return self._interpreter.SetNumThreads(i)
```
- tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc
```bash
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
```bash
  // should be the interpreter object providing the memory.
  PyObject* tensor(PyObject* base_object, int i);

  PyObject* SetNumThreads(int i);

 private:
  // Helper function to construct an `InterpreterWrapper` object.
  // It only returns InterpreterWrapper if it can construct an `Interpreter`.
```
- tensorflow/lite/tools/make/Makefile
```bash
BUILD_WITH_NNAPI=false
```
- tensorflow/lite/tools/make/targets/aarch64_makefile.inc  
**https://stackoverflow.com/questions/56055359/tensorflow-lite-arm64-error-cannot-convert-const-int8x8-t**
```bash
# Settings for generic aarch64 boards such as Odroid C2 or Pine64.
ifeq ($(TARGET),aarch64)
  # The aarch64 architecture covers all 64-bit ARM chips. This arch mandates
  # NEON, so FPU flags are not needed below.
  TARGET_ARCH := armv8-a
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-

  CXXFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -flax-vector-conversions \
    -fomit-frame-pointer \
    -fPIC

  CFLAGS += \
    -march=armv8-a \
    -funsafe-math-optimizations \
    -ftree-vectorize \
    -flax-vector-conversions \
    -fomit-frame-pointer \
    -fPIC

  LDFLAGS := \
    -Wl,--no-export-dynamic \
    -Wl,--exclude-libs,ALL \
    -Wl,--gc-sections \
    -Wl,--as-needed

       
  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl \
    -lrt

endif
```
- tensorflow/lite/build_def.bzl  
**https://github.com/tensorflow/tensorflow/issues/26731**  
**https://github.com/tensorflow/tensorflow/pull/29515/files**
```bash
            "/DTF_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
+       str(Label("//tensorflow:linux_aarch64")): [
+           "-flax-vector-conversions",
+           "-fomit-frame-pointer",
+       ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib import checkpoint
#if os.name != "nt" and platform.machine() != "s390x":
#  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib.summary import summary

if os.name != "nt" and platform.machine() != "s390x":
  try:
    from tensorflow.contrib import cloud
  except ImportError:
    pass

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
```
```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib/python3.7/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.7/dist-packages]

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
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```
```bash
$ sudo bazel build \
--config=opt \
--config=noaws \
--config=nogcp \
--config=nohdfs \
--config=noignite \
--config=nokafka \
--config=nonccl \
--local_resources=8192.0,4.0,1.0 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-flax-vector-conversions \
--copt=-fomit-frame-pointer \
//tensorflow/tools/pip_package:build_pip_package
```
```bash
$ su --preserve-environment
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# exit
$ sudo cp /tmp/tensorflow_pkg/tensorflow-1.14.0-cp37-cp37m-linux_aarch64.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-1.14.0-cp37-cp37m-linux_aarch64.whl 
```

</div></details>
  
<details><summary>Tensorflow v2.0.0-alpha</summary><div>
  
============================================================  
  
**Tensorflow v2.0.0-alpha - Stretch - Bazel 0.19.2**  

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
$ git clone -b v2.0.0-alpha0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v2.0.0-alpha0
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
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib import checkpoint
#if os.name != "nt" and platform.machine() != "s390x":
#  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib.summary import summary

if os.name != "nt" and platform.machine() != "s390x":
  try:
    from tensorflow.contrib import cloud
  except ImportError:
    pass

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
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
  
</div></details>

<details><summary>Tensorflow v2.0.0-beta0</summary><div>
  
============================================================  
  
**Tensorflow v2.0.0-beta0 - Stretch - Bazel 0.24.1**  

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
$ ./0.24.1/Raspbian_Stretch_armhf/install.sh

$ cd ~
$ git clone -b v2.0.0-beta0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v2.0.0-beta0
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
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib import checkpoint
#if os.name != "nt" and platform.machine() != "s390x":
#  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib.summary import summary

if os.name != "nt" and platform.machine() != "s390x":
  try:
    from tensorflow.contrib import cloud
  except ImportError:
    pass

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
```
- configure
```bash
$ ./configure 
Extracting Bazel installation...
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.24.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /home/pi/inference_engine_vpu_arm/python/python3.5
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
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
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
$ sudo cp /tmp/tensorflow_pkg/tensorflow-2.0.0b0-cp35-cp35m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.0.0b0-cp35-cp35m-linux_armv7l.whl 
```

</div></details>

<details><summary>Tensorflow v2.0.0-beta1</summary><div>
  
============================================================  
  
**Tensorflow v2.0.0-beta1 - Stretch - Bazel 0.24.1**  

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
$ ./0.24.1/Raspbian_Stretch_armhf/install.sh

$ cd ~
$ git clone -b v2.0.0-beta1 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v2.0.0-beta1
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
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib import checkpoint
#if os.name != "nt" and platform.machine() != "s390x":
#  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
```
- tensorflow/contrib/\_\_init\_\_.py
```python
from tensorflow.contrib.summary import summary

if os.name != "nt" and platform.machine() != "s390x":
  try:
    from tensorflow.contrib import cloud
  except ImportError:
    pass

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
```
- configure
```bash
$ ./configure 
Extracting Bazel installation...
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.24.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib
  /usr/lib/python3/dist-packages
  /home/pi/inference_engine_vpu_arm/python/python3.5
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
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=gdr         	# Build with GDR support.
	--config=verbs       	# Build with libverbs support.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=noignite    	# Disable Apache Ignite support.
	--config=nokafka     	# Disable Apache Kafka support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
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
$ sudo cp /tmp/tensorflow_pkg/tensorflow-2.0.0b1-cp35-cp35m-linux_arm7l.whl ~
```
```bash
$ cd ~
$ sudo pip3 uninstall tensorflow
$ sudo -H pip3 install tensorflow-2.0.0b1-cp35-cp35m-linux_armv7l.whl 
```

</div></details>

