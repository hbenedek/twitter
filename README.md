# Machine Learning - Project 2

## Studying Lobbying Influence in the European Parliament using Twitter data

## Abstract

Unclear relationships between politicians and lobbyists  are  a  problem  in  a  democratic  political  framework. Using  various  methods, including Graph Convolutional Networks, Logistic Regression, we  attempt  to  detect  links  between Twitter activity of Members of European Parliament and lobby groups.  The  resulting  models  are  an  improvement  over  basic text embedding  analysis.

## Team

   - Benedek Harsányi (benedek.harsanyi@epfl.ch)
   - Kamil Czerniak (kamil.czerniak@epfl.ch)
   - Mohamed Allouch (mohamed.allouch@epfl.ch)

## Running the code

In order to run the model pipeline, execute file `run.py`. However, what you may use this file for depends on your use case:

  - If you intend to run hyperparameter tuning and/or train the model for **supervised models** and evaluate them, this repository contains preprocessed data needed to do so (located in `DGL_graph.pickle`) and hence you can run these steps on your own machine.
  - If you intend to also check preprocessing or run **unsupervised model**, then please contact Aswin Suresh (aswin.suresh@epfl.ch) in order to set up a separate server environment containing raw data used when creating this model.

These precautions were taken to protect data confidentiality.

## Dependencies
  
  “Requirements.txt” is a file containing a list of items to be installed using pip install like so:

```bash
python -m pip install -r requirements.txt
```
```bash
python -m pip install git+https://github.com/facebookresearch/fastText.git
```
## Repo Structure

<pre>  
├─── run.py : final script used to replicate the predictions in our report
├─── graph.py : auxiliary functions used for graph building
├─── cross_validation.py : script for hyperparameter optimization with cross validation
├─── model.py : GCN, Logistic regression and NN model definitions and training loop
├─── preprocess.py : auxiliary functions for loading, preprocessing tha data
├─── DGL_graph.pickle : pickled deep graph object with all the embeddings
├─── clustering.py : auxiliary functions for clustering the graph and for topic discovery
├─── requirements.txt : a file containing a list of items to be installed
└─── README.md : README
</pre>