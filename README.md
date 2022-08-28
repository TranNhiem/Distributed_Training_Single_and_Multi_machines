# Distributed_Training_Single_and_Multi_machine

## Pytorch-Lightning Multi-Node training 

### Testing Pass docker Local Area Network
### network driver of container : bridge mode
### network config (cat /etc/hosts)
> master node 172.17.0.5 (container master)
> slave node 172.17.0.3 (container slave)
> all the port is avaliable
> bridge node 172.17.0.1 (docker0, special node)

### docker script :
The multi_node bash attempt to build an docker LAN, which contains two node
with each of node have 2 gpus.
> multi_node bash will be placed under the root folder of this project.

### Usage : 
At the first, ssh into the master docker with the forward port 3300
#### In local server
`ssh root@localhost -p 3300` with lab candy passwd.

#### In master container
`/opt/conda/bin/init ; source ~/.bashrc` to init conda env
`cd pytorch_lightning_distributed_training ; ./node1_bash &` to background exec
`cat /etc/hosts` to confirm the slave node ip-addr
`ssh root@172.17.0.3 -p 22` to ssh in slave node

#### In Worker container
`cd pytorch_lightning_distributed_training ; ./node2_bash &`
then the distributed learning is begin !! have fun & good luck

#### To keep the session in slave & master node, you can also install tmux or apply screen


### Pytorch 
+ Update SOON

-------------------------

### Tensorflow 


## Distributed Training on **Single Machine** 


1. Configureation and Consideration 

2. Preparing Dataset for Training 

3. Training Aggregate update Gradient

4. Training Loss update

5. Example the Training Loop for Single Machine


## Distributed Training on **Multi-Machines**

### Tensorflow 

1. Configuration and Consideration 

2. Preparing Dataset for Training Across multi-Machine 

3. Training Aggregate Gradient multi-Machine with Synchronize training 

4. Optimization (Communicate + Mixpercision Training)

5. Training loss Update 

