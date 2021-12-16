# NGAFID_Anomaly_Detection

<h1> Our Project
</h1>

<p> Our project attempts to implement several versions of multivariate anomaly detection for NGAFID flight data. Currently we are working on three seperate implementations, one a VAE model with normalizing flow, an RNN model, and a CNN model.

<h2>Setup
</h2>
<p> Please install Python, package: numpy, pandas, keras, Tensorflow. See the instruction in the sample code.

<h2>Data
</h2>
<p> Our project uses both NGAFID data as well as data from the TEP data set, available on Kaggle. (TEP: https://www.kaggle.com/averkij/tennessee-eastman-process-simulation-dataset, NGAFID: https://www.kaggle.com/hooong/ngafid-mc-20210917) 

<h2> Running our code
</h2>
<p> <br />
For RNN/CNN model, please check the NGAFID/NGAFID_Maintenance file.<br />
For Charlie-Code, standup3-Copy1.ipynb is the primary demo file to look at if looking to understand how to implement this approach.

<h2> Tennessee Eastman Process
</h2>
<p> Tennessee Eastman Process is a chemical reaction process. It is widely used for fault detection right now, given it being one of the few labeled multivariate time series data sets. <br />
