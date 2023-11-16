# NGAFID_Anomaly_Detection

<h1> Our Project
</h1>

<p> A Variational AutoEncoder used for anomaly detection on NGAFID flight data. Part of a larger project testing types of anomaly detection on this dataset, the others being a RNN based model, and a CNN based model.

<h2>Setup
</h2>
<p> Please install Python, package: numpy, pandas, keras, Tensorflow. See the instruction in the sample code.

<h2>Data
</h2>
<p> The NGAFID dataset, as well as data from a dataset known as TEP for testing (as NGAFID is unlabeled), available on Kaggle. (TEP: https://www.kaggle.com/averkij/tennessee-eastman-process-simulation-dataset, TEP: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910
/DVN/6C3JR1, NGAFID: https://www.kaggle.com/hooong/ngafid-mc-20210917) 

<h2> Running our code
</h2>
<p> <br />
For RNN/CNN model, please check the NGAFID/NGAFID_Maintenance file.<br />
For the VAE method, standup3-Copy1.ipynb is the primary demo file to look at if looking to understand how to implement this approach.

<h2> Tennessee Eastman Process
</h2>
<p> Tennessee Eastman Process is a chemical reaction process. It is widely used for fault detection right now, given it being one of the few labeled multivariate time series data sets. <br />
