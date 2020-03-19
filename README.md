# annbench: simple and lightweight benchmark for approximate nearest neighbor search

Benchmarking scripts for approximate nearest neighbor search algorithms in Python. The design of this repository is strongly influenced by a great project, [ann-benchmarks](https://github.com/erikbern/ann-benchmarks), that provides comprehensive and thorough benchmarks for various libraries. In contrast, we aim to provide more simple and lightweight benchmarks with the following features.

- Just three scripts
- Support Euclidean distance only
- Support Recall@1 only
- Support libraries that can be installed via pip/conda only
- Search with a single thread
- Sweep by a single query parameter
- Parameter specialization for each dataset
- Dynamic configuration from the command line

## [Leaderboard](https://github.com/matsui528/annbench_leaderboard)

## Getting started
```bash
git clone https://github.com/matsui528/annbench.git
cd annbench
pip install -r requirements.txt
# conda install faiss-cpu -y -c pytorch  # If you'd like to try faiss, run this on anaconda
python download.py dataset=siftsmall  # Downloaded on ./dataset
python run.py dataset=siftsmall algo=annoy  # Indices are on ./interim. Results are on ./output
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
- By default, we run each algorithm `num_trial=10` times and return the average runtime. You can change this by: `python run.py num_trial=5`

### Plot
- You can visualize the search result by `python plot.py`. This script checks `./output` and generate figures for each dataset on `./result_img`.
- As is the case in `run.py`, the same figure is written on `./log` as well.
- In order not to print query parameters, you can set the flag false: `python plot.py with_query_param=false`.
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
The class must inherit `BaseANN` class. See [annoy.py](annbench/algo/annoy.py) for examples. Then please update [proxy.py](annbench/algo/proxy.py)
- Add the name of the library on [requirements.txt](requirements.txt).
- Add a config file on `./conf/algo`. 
- Make sure the algorithm runs on a single thread
- To add a new dataset, in the same as adding a new algorithm, 
you can write a wrapper class that inherits `BaseDataset` on `./annbench/dataset`.
An simple example is  [siftsmall.py](annbench/dataset/siftsmall.py).
Don't forget to update [proxy.py](annbench/dataset/proxy.py).
- Add a config file on `./conf/dataset`.
- Feel free to send a PR!


## Advanced


### Evaluation criteria
- We followed the standard evaluation criteria in the field of computer vision, **Recall@1**. See the [evaluation function](annbench/util.py) for more details. These functions are from the [benchmark scripts of the faiss library](https://github.com/facebookresearch/faiss/tree/master/benchs).



### Index/query parameters
- We define a simple guideline to set parameters. An algorithm has to have several **index parameters** and a single **query parameter**. For one set of index parameters, one **index (data structure)** is built. For this index, we run search by sweaping the query parameter.
- For example with [ivfpq](conf/algo/ivfpq.yaml), let us consider the following index parameters:
  ```python
  param_index={"M": 8, "nlist": 100}
  ```
  With these parameters, one index (let us denote `ivfpq(M=8, nlist=100)`) is created.
  This index is stored in the disk as `M8_nlist100.bin`, where the way of naming is defined in the function [stringify_index_param](annbench/algo/ivfpq.py).
  Here, a query parameter is defined as:
  ```python
  param_query={"nprobe": [1, 2, 4, 8, 16]}
  ```
  In the search step, the index is read from the disk onto the memory first. Then we run the search five times, with `for nprobe in [1, 2, 4, 8, 16]`. This creates five results (five pairs of (recall, runtime)). By connecting these results, one polyline is drawn on the final plot.
- Note that the values of the above query parameter must be sorted. If you forget to sort (e.g., `[1, 4, 2, 8, 16]`), the final graph would become weird.

### Specialization
- Index/query parameters for each algorithm is defined in `./conf/algo/`. These parameters are used for all datasets by default. If you'd like to specialize parameters for a dataset, you can defined the specialized version in `./conf/param/`.
- For example, the default parameters for `ivfpq` is defined [here](conf/algo/ivfpq.yaml), where `nlist=100`. You can set `nlist=1000` for the sift1m dataset by adding a config file [here](conf/param/sift1m/ivfpq.yaml)


### Dynamic configuration from the command line
- In addition to editing the config files, you can override values from the commandline thanks to [hydra](https://hydra.cc/).
- Examples:
  - Writing index structures on `SOMEWHEE/LARGE_HDD/`:
    ```bash
    python run.py interim=SOMEWHERE/LARGE_HDD/interim
    ```
  - Run the `ivfpq` algorithm with different query parameters:
    ```bash
    python run.py algo=ivfpq dataset=siftsmall param_query.nprobe=[1,5,25]
    ```



## Reference
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks/)

## Author 
- [Yusuke Matsui](http://yusukematsui.me)


## Todo
- evaluate memory consumption
- default parameter
- billion-scale evaluation
