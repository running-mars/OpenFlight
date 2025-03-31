
<h1 align="center">OpenRL-Fixedwing</h1>


<h4 align="center">Less is more.</h4>

<p align="center">
  <a href="#brief-introduction">Brief Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgement">Acknowledgement</a>
</p>

## Brief Introduction

### 1. Key components:
- Rendering Engine: **Unreal Engine 4**
- Dynamics: **JSBSim** (using Skywalker X8 model)
- Middleware: **AirSim** (The last version, v1.8.1)
### 2. Multiple UAVs supported!
- **mpi4py** is used for multiple processes running together.
- **ZMQ** is used somewhere for communication between different processes or threadings.
- **Unreal Engine 5.2** can also be used, as shown in [Colosseum](https://github.com/CodexLabsLLC/Colosseum/).


## Installation

- Anaconda virtual environment is recommended. You can create a env like this (python 3.8 is recommended!):
```bash
conda create -n sim[your env name] python=3.8
```
activate it:
```bash
conda activate sim
```
- install JSBSim via pip 
```bash
pip install jsbsim
```
- install AirSim via pip (version 1.8.1)
```bash
pip install numpy pandas matplotlib simple-pid colorama
pip install msgpack-rpc-python
pip install opencv-python opencv-contrib-python
pip install airsim
```
- install pytorch if you need.
- clone and build Unreal Engine refer to AirSim/Colosseum
- clone and build AirSim/Colosseum (please note the version)
- configure the UE project and the AirSim Plugin.

## How To Use

Run the training script using Pycharm or command line:
```bash
python navp2p_v1_train.py
```
When validation, run:
```bash
python navp2p_v1_test.py
```
or:
```bash
python navp2p_v1_test_multi.py
```

## License

MIT


---
## Acknowledgement

This open-source project has referenced and drawn inspiration from many other open-source projects. 
For the purpose of organizing the code structure and making it clearer, we did not directly fork these repositories, but instead, organized, improved, and added new features in a new repository. 
The code in the repository is not entirely written by us. We extend our respect and gratitude to the developers of the following open-source projects and release this project as open-source.

1. [Fixedwing-Airsim](https://github.com/AOS55/Fixedwing-Airsim)

2. [gym-jsbsim](https://github.com/Gor-Ren/gym-jsbsim)

3. [AirSim](https://github.com/microsoft/AirSim)

4. [Colosseum](https://github.com/CodexLabsLLC/Colosseum)

5. [dwa](https://github.com/estshorter/dwa)