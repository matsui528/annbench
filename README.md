# annbench: a lightweight benchmark for approximate nearest neighbor search

`annbench` is a simple benchmark for approximate nearest neighbor search algorithms in Python. This repository design is strongly influenced by a great project, [ann-benchmarks](https://github.com/erikbern/ann-benchmarks), that provides comprehensive and thorough benchmarks for various algorithms. In contrast, we aim to deliver more lightweight and straightforward scripts with the following features.

- Support Euclidean distance only
- Support Recall@1 only
- Support libraries installable via pip/conda only
- Search with a single thread
- Sweep by a single query parameter

## [Leaderboard](https://github.com/matsui528/annbench_leaderboard)
![](https://github.com/matsui528/annbench_leaderboard/blob/main/result_img/2021_02_21/sift1m.png?raw=True)


## Getting started
```bash
git clone https://github.com/matsui528/annbench.git
cd annbench
pip install -r requirements.txt
# conda install faiss-cpu -y -c pytorch  # If you'd like to try faiss, run this on anaconda
# conda install faiss-gpu -y -c pytorch  # or, if you have GPUs, install faiss-gpu

python download.py dataset=siftsmall  # Downloaded on ./dataset

python run.py dataset=siftsmall algo=annoy  # Indices are on ./interim. Results are on ./output

python plot.py   # Plots are on ./result_img
```

## Run all algorithms on all dataset
```bash
# Downloading deep1m takes some hours
python download.py --multirun dataset=siftsmall,sift1m,deep1m

# Will take some hours
python run.py --multirun dataset=siftsmall,sift1m,deep1m algo=linear,annoy,ivfpq,hnsw
# Or, if you have GPUs, 
# python run.py --multirun dataset=siftsmall,sift1m,deep1m algo=linear,annoy,ivfpq,hnsw,linear_gpu,ivfpq_gpu

python plot.py
```

## How it works

### Config
- All config files are on `./conf`. You can edit the config files to change parameters.

### Download
- Download a target dataset by `python download.py dataset=DATASET`. 
Several datasets can be downloaded at once by `python download.py --multirun dataset=DATASET1,DATASET2,DATASET3`. See [hydra](https://hydra.cc/) for more detailed APIs for multirun.
- The data will be placed on `./dataset`.

### Run
- Evaluate a target algorithm (`ALGO`) with a target dataset (`DATASET`) by `python run.py dataset=DATASET algo=ALGO`. You can run multiple algorithms on multiple datasets by `python run.py --multirun dataset=DATASET1,DATASET2 algo=ALGO1,ALGO2`.
- Indices (data structures for search) are stored on `./interim`. They are reused for each run with different query parameters.
- The search result will be on `./output`.
- By default, we run each algorithm `num_trial=10` times and return the average runtime. You can change this by: `python run.py num_trial=5`

### Plot
- You can visualize the search result by `python plot.py`. This script checks `./output` and generate figures for each dataset on `./result_img`.
- In order not to print query parameters, you can set the flag false: `python plot.py with_query_param=false`.
- To change the size of the image: `python plot.py width=15 height=10`


### Log
- When running `run.py` or `plot.py`, the output files will be on `./log` as well. For example with `python run.py algo=annoy dataset=siftsmall`, the result file will be saved on (1) `./output/siftsmall/annoy/result.yaml` and (2) `./log/2020-03-11/22-30-59/0/result.yaml`.

## Supported datasets
| dataset | dim | #base | #query | #train | misc
| --- | --- | --- | --- | --- | --- |
| siftsmall | 128 |    10,000 |    100 |  25,000 | A toy dataset for hello-world|
| sift1m    | 128 | 1,000,000 | 10,000 | 100,000 | |
| deep1m    |  96 | 1,000,000 | 10,000 | 100,000 | The first 1M vectors of [Deep1B](https://github.com/arbabenko/GNOIMI). [Hepler scripts](https://github.com/matsui528/deep1b_gt)|

## Supported Algorithms
- [linear scan (faiss)](https://github.com/facebookresearch/faiss/blob/master/faiss/IndexFlat.h)
- [ivfpq (faiss)](https://github.com/facebookresearch/faiss/blob/master/faiss/IndexIVFPQ.h)
- [ivfpq with 4-bit pq (faiss)](https://github.com/facebookresearch/faiss/blob/master/faiss/IndexIVFPQFastScan.h)
- [annoy](https://github.com/spotify/annoy)
- [hnsw](https://github.com/nmslib/hnswlib)



## Add a new algorithm
- Write a wrapper class on `./annbench/algo`. 
The class must inherit `BaseANN` class. See [annoy.py](annbench/algo/annoy.py) for examples.
- Update [./annbench/algo/proxy.py](annbench/algo/proxy.py)
- Add the name of the library on [requirements.txt](requirements.txt).
- Add a config file on `./conf/algo`. 
- Make sure the algorithm runs on a single thread

## Add a new dataset
- Write a wrapper class that inherits `BaseDataset` on `./annbench/dataset`.
An simple example is  [siftsmall.py](annbench/dataset/siftsmall.py).
- Update [./annbench/dataset/proxy.py](annbench/dataset/proxy.py).
- Add a config file on `./conf/dataset`.


## Advanced


### Evaluation criteria
- We followed the standard evaluation criteria in the field of computer vision, **Recall@1**. See the [evaluation function](annbench/util.py) for more details. These functions are from the [benchmark scripts of the faiss library](https://github.com/facebookresearch/faiss/tree/master/benchs).



### Index/query parameters
- We define a simple guideline to set parameters. An algorithm has to have several **index parameters** and a single **query parameter**. For one set of index parameters, one **index (data structure)** is built. For this index, we run the search by sweeping the query parameter.
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
- Note that you must sort the values of the above query parameter. If you forget to sort (e.g., `[1, 4, 2, 8, 16]`), the final graph would become weird.

### Specialization
- Index/query parameters for each algorithm is defined in `./conf/algo/`. These parameters are used for all datasets by default. If you'd like to specialize parameters for a specific dataset, you can define the specialized version in `./conf/specialized_param/`.
- For example, the default parameters for `ivfpq` is defined [here](conf/algo/ivfpq.yaml), where `nlist=100`. You can set `nlist=1000` for the sift1m dataset by adding a config file [here](conf/specialized_param/sift1m_ivfpq.yaml)


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

## Contribute
- Feel free to open a pull request

## Author 
- [Yusuke Matsui](http://yusukematsui.me)
