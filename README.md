# anomaly-detection-log-datasets

## Getting the data sets

### HDFS

Fore more information on this data set, see
 * Xu, W., Huang, L., Fox, A., Patterson, D., & Jordan, M. I. (2009, October). Detecting large-scale system problems by mining console logs. In Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles (pp. 117-132).

```
wget http://iiis.tsinghua.edu.cn/~weixu/demobuild.zip
unzip demobuild.zip
gunzip -c data/online1/lg/sorted.log.gz > sorted.log
```

### BGL

Fore more information on this data set, see
 * Oliner, A., & Stearley, J. (2007, June). What supercomputers say: A study of five system logs. In 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN'07) (pp. 575-584). IEEE.

```
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz
gunzip bgl2.gz
```

### OpenStack

Fore more information on this data set, see
 * Du, M., Li, F., Zheng, G., & Srikumar, V. (2017, October). Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security (pp. 1285-1298).

```
wget https://zenodo.org/record/3227177/files/OpenStack.tar.gz
tar -xvf OpenStack.tar.gz
```

### Hadoop

For more information on this data set, see
 * Lin, Q., Zhang, H., Lou, J. G., Zhang, Y., & Chen, X. (2016, May). Log clustering based problem identification for online service systems. In Proceedings of the 38th International Conference on Software Engineering Companion (pp. 102-111).

```
wget https://zenodo.org/record/3227177/files/Hadoop.tar.gz
mkdir logs
tar -xvf Hadoop.tar.gz -C logs
```

### ADFA

For more information on this data set, see
 * Creech, G., & Hu, J. (2013, April). Generation of a new IDS test dataset: Time to retire the KDD collection. In 2013 IEEE Wireless Communications and Networking Conference (WCNC) (pp. 4487-4492). IEEE.

```
git clone https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset
unzip a-labelled-version-of-the-ADFA-LD-dataset/ADFA-LD.zip -d .
```

If you use resources from this repository, please cite the following publication:
* Landauer, M., Onder, S., Skopik, F., & Wurzenberger, M. (2023): [Deep Learning for Anomaly Detection in Log Data: A Survey](https://arxiv.org/abs/2207.03820). arXiv preprint arXiv:2207.03820. \[[PDF](https://arxiv.org/pdf/2207.03820.pdf)\]
