# anomaly-detection-log-datasets

## Getting the data sets

### HDFS

```
wget http://iiis.tsinghua.edu.cn/~weixu/demobuild.zip
unzip demobuild.zip
gunzip -c data/online1/lg/sorted.log.gz > sorted.log
```

Fore more information on this data set, see
 * Xu, W., Huang, L., Fox, A., Patterson, D., & Jordan, M. I. (2009, October). Detecting large-scale system problems by mining console logs. In Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles (pp. 117-132).

### BGL

```
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz
gunzip bgl2.gz
```

Fore more information on this data set, see
 * Oliner, A., & Stearley, J. (2007, June). What supercomputers say: A study of five system logs. In 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN'07) (pp. 575-584). IEEE.

### OpenStack

```
wget https://zenodo.org/record/3227177/files/OpenStack.tar.gz
tar -xvf OpenStack.tar.gz
```

Fore more information on this data set, see
 * Du, M., Li, F., Zheng, G., & Srikumar, V. (2017, October). Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security (pp. 1285-1298).

### Hadoop

```
wget https://zenodo.org/record/3227177/files/Hadoop.tar.gz
mkdir logs
tar -xvf Hadoop.tar.gz -C logs
```

For more information on this data set, see
 * Lin, Q., Zhang, H., Lou, J. G., Zhang, Y., & Chen, X. (2016, May). Log clustering based problem identification for online service systems. In Proceedings of the 38th International Conference on Software Engineering Companion (pp. 102-111).

### Thunderbird

```
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz
gunzip tbird2.gz
```

Fore more information on this data set, see
 * Oliner, A., & Stearley, J. (2007, June). What supercomputers say: A study of five system logs. In 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN'07) (pp. 575-584). IEEE.

### ADFA

```
git clone https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset
unzip a-labelled-version-of-the-ADFA-LD-dataset/ADFA-LD.zip -d .
```

For more information on this data set, see
 * Creech, G., & Hu, J. (2013, April). Generation of a new IDS test dataset: Time to retire the KDD collection. In 2013 IEEE Wireless Communications and Networking Conference (WCNC) (pp. 4487-4492). IEEE.

## Parsing the data sets

To parse the data, run the respective <dataset>_parse.py script. The templates used for parsing are taken from [Logpai/Logparser](https://github.com/logpai/logparser) and adapted or extended to make sure that all logs are parsed and that each log event only fits into to one template.

```
python3 hdfs_parse.py
```

## Sampling the data sets
  
To generate training and test files, run the sample.py script and specify the directory of the log data to be sampled. Moreover, the sampling ratio can be specified. For example, use the following command to sample the HDFS log data set so that the training file comprises 1% of the normal events.
  
```
python3 sample.py --data_dir hdfs_xu --train_ratio 0.01
```
  
## Citation

If you use any resources from this repository, please cite the following publication:
* Landauer, M., Onder, S., Skopik, F., & Wurzenberger, M. (2023): [Deep Learning for Anomaly Detection in Log Data: A Survey](https://arxiv.org/abs/2207.03820). arXiv preprint arXiv:2207.03820. \[[PDF](https://arxiv.org/pdf/2207.03820.pdf)\]
