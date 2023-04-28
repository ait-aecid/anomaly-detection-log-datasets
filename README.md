# anomaly-detection-log-datasets

## Getting the data sets

### HDFS

```
wget http://iiis.tsinghua.edu.cn/~weixu/demobuild.zip
unzip demobuild.zip
gunzip -c data/online1/lg/sorted.log.gz > sorted.log
```

### BGL

```
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz
gunzip bgl2.gz
```

### OpenStack

```
wget https://zenodo.org/record/3227177/files/OpenStack.tar.gz
tar -xvf OpenStack.tar.gz
```

### Hadoop

```
wget https://zenodo.org/record/3227177/files/Hadoop.tar.gz
mkdir logs
tar -xvf Hadoop.tar.gz -C logs
```

### ADFA

```
git clone https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset
unzip a-labelled-version-of-the-ADFA-LD-dataset/ADFA-LD.zip -d .
```

If you use resources from this repository, please cite the following publication:
* Landauer, M., Onder, S., Skopik, F., & Wurzenberger, M. (2023): [Deep Learning for Anomaly Detection in Log Data: A Survey](https://arxiv.org/abs/2207.03820). arXiv preprint arXiv:2207.03820. \[[PDF](https://arxiv.org/pdf/2207.03820.pdf)\]
