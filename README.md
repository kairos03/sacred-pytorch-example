# Sacred Example for Pytorch
This is Experiment logging tool [**sacred**](https://github.com/IDSIA/sacred) example for pytorch

# Quick Start
##  1. Modify `config.py` file.
Replace the const(MONGODB_URI, MONGO_DB) for yours.

```python
# config.py
MONGO_URI = 'mongodb://USER:PASSWORD@MONGO_IP:27017'
MONGO_DB = 'DBNAME'
```

## 2. run train.py
Run the `train.py` to start your experiment!

```bash
hyperparam -  Namespace(batch_size=512, epochs=3, gamma=0.7, log_interval=50, lr=1.0, no_cuda=False, save_model=False, test_batch_size=1000, workers=1)
INFO - pytorch-mnist - Running command 'main'
INFO - pytorch-mnist - Started run with ID "15"
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.330945
Train Epoch: 1 [25600/60000 (0%)]       Loss: 0.366348
Train Epoch: 1 [51200/60000 (0%)]       Loss: 0.185399

Test 1 Average loss: 0.0002, Accuracy: 9710/10000 (97%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.114014
Train Epoch: 2 [25600/60000 (0%)]       Loss: 0.108499
Train Epoch: 2 [51200/60000 (0%)]       Loss: 0.103628

Test 2 Average loss: 0.0001, Accuracy: 9846/10000 (98%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.042696
Train Epoch: 3 [25600/60000 (0%)]       Loss: 0.062931
Train Epoch: 3 [51200/60000 (0%)]       Loss: 0.100096

Test 3 Average loss: 0.0001, Accuracy: 9870/10000 (99%)

INFO - pytorch-mnist - Result: 98.7
INFO - pytorch-mnist - Completed after 0:00:36
```

## 3. (option) See with Omniboard
You can see the experiment's result with [Omniboard](https://github.com/vivekratnavel/omniboard)
![omniboard](image/omniboard.png)
