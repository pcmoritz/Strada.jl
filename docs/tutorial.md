# Tutorial

In this tutorial, we show how to get started with Strada and run it with the GPU backend.

## Setting up Amazon's GPU instances

These steps are from [the Caffe wiki](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-%28Ubuntu,-CUDA-7,-cuDNN%29). First, start one of Amazon's GPU instances (g2.2xlarge or g2.8xlarge) using Ubuntu 14.04 64 bit as the AMI. We also recommend to increase the root `/dev/sda1` device size to something larger than 8 GiB.

First update the system and install `build-essential`:

```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential
```

Next, download the NVIDIA driver

```bash
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
```

Extract the installers using

```bash
chmod +x cuda_7.0.28_linux.run
mkdir nvidia_installers
./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
```

Then update the linux image to be compatible with NVIDIA's drivers:

```bash
sudo apt-get install linux-image-extra-virtual
```

While installing the linux-image-extra-virtual, you may be prompted "What would you like to do about menu.lst?". I selected "keep the local version currently installed".

Now we have to disable nouveau since it conflicts with NVIDIA's kernel module. Open

```bash
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

and add the following lines to this file:

```bash
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

Back in the shell, execute the commands:

```bash
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot
```

After the reboot, we can finally install the driver:

```bash
sudo apt-get install linux-source
sudo apt-get install linux-headers-`uname -r`

cd nvidia_installers
sudo ./NVIDIA-Linux-x86_64-346.46.run
```

Just select the defaults for all the questions that pop up.

Now we can install CUDA

```bash
sudo modprobe nvidia
sudo ./cuda-linux64-rel-7.0.28-19326674.run
sudo ./cuda-samples-linux-7.0.28-19326674.run
```

Follow the suggestion to add the following to your `.bashrc`

```bash
export PATH=$PATH:/usr/local/cuda-7.0/bin
export LD_LIBRARY_PATH=:/usr/local/cuda-7.0/lib64
```

and activate it by running `source ~/.bashrc`.

## Installing Julia and Strada

Install Julia with

```bash
sudo add-apt-repository ppa:staticfloat/juliareleases
sudo add-apt-repository ppa:staticfloat/julia-deps
sudo apt-get update
sudo apt-get install julia
```

To install Strada, call `julia` and run

```julia
Pkg.clone("https://github.com/pcmoritz/Strada.jl")
Pkg.build("Strada")
```

## Trying out models

First, download the MNIST data set from Yann LeCun's website by running the commands

```bash
cd ~/.julia/v0.3/Strada/data/
bash get-mnist.sh
```

Now you can train the model on the CPU by running

```bash
cd ~/.julia/v0.3/Strada/examples
julia train-mnist.jl
```

To train the models on a GPU, you should open `train-mnist.jl` and uncomment the line `set_gpu_mode(net)`.
