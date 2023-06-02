# anomaly-detection-log-datasets

## Getting the data sets

### HDFS

There are three versions of this data set: The original logs from Wei Xu et al., and two alternative versions from Loghub and LogDeep.

#### Original logs from Wei Xu et al.

The original logs can be retrieved from (Wei Xu's website)[http://people.iiis.tsinghua.edu.cn/~weixu/sospdata.html] as follows.

```shell
cd hdfs_xu/
wget http://iiis.tsinghua.edu.cn/~weixu/demobuild.zip
unzip demobuild.zip
gunzip -c data/online1/lg/sorted.log.gz > sorted.log
```

Fore more information on this data set, see
 * Xu, W., Huang, L., Fox, A., Patterson, D., & Jordan, M. I. (2009, October). Detecting large-scale system problems by mining console logs. In Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles (pp. 117-132).

#### Loghub

Another version of this data set is provided by (Loghub)[https://github.com/logpai/loghub]. Note that some lines appear to be missing from the original logs.

```shell
cd hdfs_loghub/
wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz
tar -xvf HDFS_1.tar.gz 
```

For more information on Loghub, see
 * He, S., Zhu, J., He, P., & Lyu, M. R. (2020). Loghub: a large collection of system log datasets towards automated log analytics. arXiv preprint arXiv:2008.06448.

#### LogDeep

A pre-processed version of the data set is provided in the (LogDeep repository)[https://github.com/donglee-afar/logdeep]. Note that timestamps and sequence identifiers are missing from this data set.

```shell
cd hdfs_logdeep/
git clone https://github.com/donglee-afar/logdeep.git
mv logdeep/data/hdfs/hdfs_t* .
```

### BGL

There are two versions of this data set: CFDR and Loghub.

#### CFDR

The original logs from the (Computer Failure Data Repository)[https://www.usenix.org/cfdr] can be retrieved as follows.

```shell
cd bgl_cfdr/
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz
gunzip bgl2.gz
```

Fore more information on this data set, see
 * Oliner, A., & Stearley, J. (2007, June). What supercomputers say: A study of five system logs. In 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN'07) (pp. 575-584). IEEE.

#### Loghub

An alternative version of this data set is provided by (Loghub)[https://github.com/logpai/loghub]. Note that some logs have different labels in this data set.

```shell
cd bgl_loghub/
wget https://zenodo.org/record/3227177/files/BGL.tar.gz
tar -xvf BGL.tar.gz
```

For more information on Loghub, see
 * He, S., Zhu, J., He, P., & Lyu, M. R. (2020). Loghub: a large collection of system log datasets towards automated log analytics. arXiv preprint arXiv:2008.06448.

### OpenStack

The original OpenStack logs are not available anymore; however, (Loghub)[https://github.com/logpai/loghub] provides a version of this data set.

```shell
cd openstack_loghub/
wget https://zenodo.org/record/3227177/files/OpenStack.tar.gz
tar -xvf OpenStack.tar.gz
```

Fore more information on this data set, see
 * Du, M., Li, F., Zheng, G., & Srikumar, V. (2017, October). Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security (pp. 1285-1298).
 * He, S., Zhu, J., He, P., & Lyu, M. R. (2020). Loghub: a large collection of system log datasets towards automated log analytics. arXiv preprint arXiv:2008.06448.

### Hadoop

The original Hadoop logs are not available anymore; however, (Loghub)[https://github.com/logpai/loghub] provides a version of this data set.

```shell
cd hadoop_loghub/
wget https://zenodo.org/record/3227177/files/Hadoop.tar.gz
mkdir logs
tar -xvf Hadoop.tar.gz -C logs
```

For more information on this data set, see
 * Lin, Q., Zhang, H., Lou, J. G., Zhang, Y., & Chen, X. (2016, May). Log clustering based problem identification for online service systems. In Proceedings of the 38th International Conference on Software Engineering Companion (pp. 102-111).
 * He, S., Zhu, J., He, P., & Lyu, M. R. (2020). Loghub: a large collection of system log datasets towards automated log analytics. arXiv preprint arXiv:2008.06448.

### Thunderbird

The original logs from the (Computer Failure Data Repository)[https://www.usenix.org/cfdr] can be retrieved as follows.

```shell
cd thunderbird_cfdr/
wget http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz
gunzip tbird2.gz
```

For more information on this data set, see
 * Oliner, A., & Stearley, J. (2007, June). What supercomputers say: A study of five system logs. In 37th annual IEEE/IFIP international conference on dependable systems and networks (DSN'07) (pp. 575-584). IEEE.

### ADFA

The (verazuo repository)[https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset] provides a labeled version of this data set.

```shell
cd adfa_verazuo/
git clone https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset
unzip a-labelled-version-of-the-ADFA-LD-dataset/ADFA-LD.zip -d .
```

For more information on this data set, see
 * Creech, G., & Hu, J. (2013, April). Generation of a new IDS test dataset: Time to retire the KDD collection. In 2013 IEEE Wireless Communications and Networking Conference (WCNC) (pp. 4487-4492). IEEE.

## Parsing the data sets

To parse the data, run the respective `<dataset>_parse.py` script. For example, use the following command to parse the HDFS log data set:

```shell
python3 hdfs_parse.py
```

The templates used for parsing are taken from [Logpai/Logparser](https://github.com/logpai/logparser) and adapted or extended to make sure that all logs are parsed and that each log event only fits into to one template. The Thunderbird log data set is an exception; due to the complexity and length of the data set, we used our [aecid-incremental-clustering](https://github.com/ait-aecid/aecid-incremental-clustering) and [aecid-parsergenerator](https://github.com/ait-aecid/aecid-parsergenerator) to generate event templates, however, some of them are overly specific or generic and log lines may match multiple events.

## Sampling the data sets
  
To generate training and test files, run the sample.py script and specify the directory of the log data to be sampled. Moreover, the sampling ratio can be specified. For example, use the following command to sample the HDFS log data set so that the training file comprises 1% of the normal events.
  
```shell
python3 sample.py --data_dir hdfs_xu --train_ratio 0.01
```

This will generate the files `<dataset>_train`, `<dataset>_test_normal`, and `<dataset>_test_abnormal` in the respective directory. In case that fine-granular anomaly labels are available, the sampling script will also generate `<dataset>_test_abnormal_<anomaly>`, which contain only those sequences that correspond to the respective anomaly class.
  
## Analyzing the data sets

Run the analysis script to output some basic information about the data sets, specifically regarding the distributions of normal and anomalous samples. The script will also show the most frequent normal and anomalous sequences and count vectors; the number of displayed samples is specified with the `show_samples` parameter.

```shell
python3 analyze.py --data_dir hdfs_xu --show_samples 3
Load parsed sequences ...
Parsed lines total: 12580989
Parsed lines normal: 12255230 (97.4%)
Parsed lines anomalous: 325759 (2.6%)
Event types total: 33
Event types normal: 20 (60.6%)
Event types anomalous: 32 (97.0%)
Sequences total: 575061
Sequences normal: 558223 (97.1%)
Sequences anomalous: 16838 (2.9%)
Unique sequences: 26814 (4.7%)
Unique sequences normal: 21690 (80.9%)
Unique sequences anomalous: 5133 (19.1%)
Sequences labeled normal that also occur as anomalous: 14 (0.003%), 9 unique
Sequences labeled anomalous that also occur as normal: 17 (0.101%), 9 unique
Common normal sequences:
73691: ('5', '5', '5', '22', '11', '9', '11', '9', '11', '9', '26', '26', '26', '23', '23', '23', '30', '30', '30', '21', '21', '21')
40156: ('5', '22', '5', '5', '11', '9', '11', '9', '11', '9', '26', '26', '26', '23', '23', '23', '30', '30', '30', '21', '21', '21')
37788: ('5', '5', '22', '5', '11', '9', '11', '9', '11', '9', '26', '26', '26', '23', '23', '23', '30', '30', '30', '21', '21', '21')
Common anomalous sequences:
1643: ('5', '22')
1361: ('22', '5', '5', '7')
1307: ('22', '5')
Unique count vectors: 666 (2.5% of unique sequences or 0.1% of all sequences)
Unique count vectors normal: 257 (38.6%)
Unique count vectors anomalous: 418 (62.8%)
Count vectors labeled normal that also occur as anomalous: 230 (0.041%), 9 unique
Count vectors labeled anomalous that also occur as normal: 316 (1.877%), 9 unique
Common normal count vectors:
300011: (('11', 3), ('21', 3), ('22', 1), ('23', 3), ('26', 3), ('30', 3), ('5', 3), ('9', 3))
96316: (('11', 3), ('22', 1), ('26', 3), ('5', 3), ('9', 3))
21127: (('11', 3), ('21', 3), ('22', 1), ('23', 3), ('26', 3), ('3', 1), ('30', 3), ('4', 2), ('5', 3), ('9', 3))
Common anomalous count vectors:
3225: (('22', 1), ('5', 2), ('7', 1))
3182: (('11', 3), ('20', 1), ('21', 3), ('22', 1), ('23', 3), ('26', 3), ('30', 3), ('5', 3), ('9', 3))
2950: (('22', 1), ('5', 1))
```

## Citation

If you use any resources from this repository, please cite the following publications:
* Landauer, M., Skopik, F., & Wurzenberger, M.: A Critical Review of Common Log Data Sets Used for Evaluation of Sequence-based Anomaly Detection Techniques. Under Review.
* Landauer, M., Onder, S., Skopik, F., & Wurzenberger, M. (2023): [Deep Learning for Anomaly Detection in Log Data: A Survey](https://www.sciencedirect.com/science/article/pii/S2666827023000233). Machine Learning with Applications, Volume 12, 15 June 2023, 100470, Elsevier. \[[PDF](https://arxiv.org/pdf/2207.03820.pdf)\]
