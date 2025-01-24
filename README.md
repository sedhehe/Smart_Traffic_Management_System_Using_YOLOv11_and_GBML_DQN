# Smart Traffic Management System
This my major project where we control the traffic signal based on the traffic density and prioritizing emergency vehicles if present using GBML-DQN and detect vehicles using YOLOv11

## To run
first install the requirements
```
pip install -r requirements.txt
```
## Install pytorch
install DQN rl using pytorch based on your device, mine doesn't have gpu so i installed cpu version if u have cpu download the CUDA version
https://pytorch.org/get-started/locally/

## Install sumo
install sumo simulator from https://eclipse.dev/sumo/

## To Train
```
python main.py
```

### To Test
```
python test.py
```