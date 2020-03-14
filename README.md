# annbench: simple and lightweight benchmark for approximate nearest neighbor search

Benchmarking scripts for approximate nearest neighbor search algorithms in Python. The design of this repository is strongly influenced by a great project, [ann-benchmarks](https://github.com/erikbern/ann-benchmarks), that provides comprehensive and thorough benchmarks for various libraries. In contrast, we aim to provide more simple and lightweight benchmarks with the following features.

- Just three scripts
- Support Euclidean distance only
- Support Recall@1 only
- Support libraries that can be installed via pip/conda only
- Sweep by a single query parameter
- Seach with a single thread

## [Leaderboard](https://github.com/matsui528/annbench_leaderboard)

## Getting started
```bash
git clone https://github.com/matsui528/annbench.git
cd annbench
pip install -r requirements.txt
# conda install faiss-cpu -y -c pytorch  # If you'd like to try faiss, run this on anaconda
python download.py dataset=siftsmall  # Donloaded on ./dataset
python run.py dataset=siftsmall algo=annoy  # Indexes are on ./interim. Results are on ./output
python plot.py   # Plots are on ./result_img
```

## Run all algorithms on all dataset
- It may take some hours for building. Once the indices are built, the search takes some minutes.
```bash
python download.py --multirun dataset=siftsmall,sift1m
python run.py --multirun dataset=siftsmall,sift1m algo=linear,annoy,ivfpq,hnsw
python plot.py
```

## How it works

### Config
- All config files are on `./conf`. You can edit config files to change parameters.

### Download
- Download a target dataset by `python download.py dataset=DATASET`. 
Several datasets can be downloaded at once by `python download.py --multirun dataset=DATASET1,DATASET2,DATASET3`. See [hydra](https://hydra.cc/) for more detailed APIs for multirun.
- The data will be placed on `./dataset`. The logs are written on `./log`.

### Run
- Evaluate a target algorithm (`ALGO`) with a target dataset (`DATASET`) by `python run.py dataset=DATASET algo=ALGO`. You can run multiple algorithms on multiple datasets by `python run.py --multirun dataset=DATASET1,DATASET2 algo=ALGO1,ALGO2`.
- Indices (data structures for search) are stored on `./interim`. They are reused for each run with different query parameters.
- The search result will be on `./output`. The same file will be on `./log` as well. For example with `algo=annoy` and `dataset=siftsmall`, the result file is `./output/siftsmall/annoy/result.yaml`, and it is identical to something like `./log/2020-03-11/22-30-59/0/result.yaml`.

### Plot
- You can visualize the search result by `python plot.py`. This script checks `./output` and generate figures for each dataset on `./result_img`.
- In order not to print query parameters, set the flag false: `python plot.py with_query_param=false`.
- To change the size of the image: `python plot.py width=15 height=10`





## Supported datasets
| dataset | dimension | #base | #query | #train | misc
| --- | --- | --- | --- | --- | --- |
| siftsmall | 128 |    10,000 |    100 |  25,000 | A toy dataset for hello-world|
| sift1m    | 128 | 1,000,000 | 10,000 | 100,000 | |

## Supported Algorithms
- [linear scan (faiss)](https://github.com/facebookresearch/faiss)
- [ivfpq (faiss)](https://github.com/facebookresearch/faiss)
- [annoy](https://github.com/spotify/annoy)
- [hnsw](https://github.com/nmslib/hnswlib)



## Add new algorithms/datasets
- To add a new algorithm, please write a wrapper class on `./annbench/algo`. 
The class must inherit `BaseANN` class. See [annoy.py](annbench/algo/annoy.py) for examples. Finally, update [proxy.py](annbench/algo/proxy.py)
- Add the name of the library on [requirements.txt](requirements.txt).
- Make sure the algorithm runs on a single thread
- To add a new dataset, in the same as adding a new algorithm, 
you can write a wrapper class that inherits `BaseDataset` on `./annbench/dataset`.
An simple example is  [siftsmall.py](annbench/dataset/siftsmall.py).
Don't forget to update [proxy.py](annbench/dataset/proxy.py).
- Feel free to send a PR!


## Advanced


### Evaluation criteria
- We followed the standard evaluation criteria in the field of computer vision, **Recall@1**. See the [evaluation function](annbench/util.py) for more details. These functions are from the [benchmark scripts of the faiss library](https://github.com/facebookresearch/faiss/tree/master/benchs).



### Index/query parameters
- We define a simple guideline to set parameters. An algorithm has to have several **index parameters** and a single **query parameter**. For one set of index parameters, one **index (data structure)** is built. For this index, we run search by sweaping the query parameter.
- For example with [ivfpq](conf/algo/ivfpq.yaml), let us consider the following index parameters:
  ```python
  param_index={"M": 4, "nlist": 100, "filename": "M4_nlist100.bin"}
  ```
  With these parameters, one index (let us denote `ivfpq(M=4, nlist=100)`) is created, and will be stored as `M4_nlist100.bin`. 
  Here, a query parameter is defined as:
  ```python
  param_query=[{"nprobe": 1}, {"nprobe": 4}, {"nprobe": 16}]
  ```
  In the search step, the index is read from the disk onto the memory first. Then we run the search three times, with `for nprobe in [1, 4, 16]`. This creates three results (three pairs of (recall, runtime)). By connecting these results, one polyline is drawn on the final plot.
- We don't exhaustively evaluate all parameter combinations (e.g., M={4,8,16} * nlist={100, 200, 400} could result in nine indices). Instead, we evaluate some typical parameter configurations only, e.g., {M=4, nlist=100}, {M=8, nlist=100}, and so on. This is because nobody can check all parameter combinations for their own large-scale data in the real world scenario. We believe the scores with typical parameter settings are more important. A good algorithm should achieve a good result with the default parameter.

### Dynamic configuration from the command line
- In addition to editing the config files, you can override values from the commandline thanks to [hydra](https://hydra.cc/), e.g, `python run.py interim=SOMEWHERE/LARGEHDD/interim`.


### Launch on AWS
- Run the following commands on a fresh AWS EC2 instance to reproduce the result
  ```bash
  sudo apt -y update
  sudo apt -y upgrade
  sudo apt -y install build-essential
  wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O $HOME/anaconda.sh  
  bash $HOME/anaconda.sh -b -p $HOME/anaconda
  echo 'export PATH="$HOME/anaconda/bin:$PATH"' >> $HOME/.bashrc
  source $HOME/.bashrc
  conda update conda --yes
  conda update --all --yes
  ```




## Reference
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks/)

## Author 
- [Yusuke Matsui](http://yusukematsui.me)


## Todo
- evaluate memory consumption
- default parameter
- run multiple times and take an average
- billion-scale evaluation
