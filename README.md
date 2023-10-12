# DistributionsForFL
This repository implements common ways of splitting a dataset to create non-i.i.d. federated learning distributions.

## Splits
* iid: distributed i.i.d. between clients with equal sample sizes
* shard: based on the strategy outlined in [Communication-Efficient Learning of Deep Networks from 
  Decentralized Data (McMahan et. al.)](https://arxiv.org/abs/1602.05629). Generalized to let the user specify the 
  number of shard received by each client
* Dirichlet equal: Uses a Dirichlet distribution parameterized by alpha [0.01, infinity) and partitions samples so that
  clients have an equal number of samples
* Dirichlet unequal: Uses a Dirichlet distribution parameterized by alpha [0.01, infinity) and partitions samples so that
  clients have an unequal number of samples (Not implemented yet)
  
### Sample Splits
Sample bar graphs for the proportions of CIFAR10 classes of 3 randomly selected clients, parameterized by alpha=0.1. L1, L2, ..., L10 along the y-axis are the class labels, the x-axis are the proportions of each class s.t. L1 + L2 + ... + L10 = 1.0

|Client No.|Proportion|Visualization|
|----------|----------|-------------|
|10|[0.    0.    0.288 0.    0.    0.    0.    0.    0.712 0.   ]|![alt text](https://github.com/GwenLegate/DistributionsForFL/blob/main/propImages/prop10.png?raw=true)|
|4|[0.     0.0028 0.1088 0.     0.     0.5692 0.3172 0.     0.002  0.    ]|![alt text](https://github.com/GwenLegate/DistributionsForFL/blob/main/propImages/prop4.png?raw=true)|
|5|[0.     0.3848 0.0084 0.0056 0.     0.1204 0.3592 0.1192 0.0024 0.    ]|![alt text](https://github.com/GwenLegate/DistributionsForFL/blob/main/propImages/prop5.png?raw=true)|
  
## Use of Code
Install requirements and execute `python main.py`. You can set the type of split, number of users and alpha for the 
Dirichlet distribution in main.py 
