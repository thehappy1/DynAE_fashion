# Deep Clustering with a Dynamic Autoencoder:  From Reconstruction towards Centroids Construction


# Abstract

In unsupervised learning, there is no apparent straightforward cost function that can capture the significant factors of variations and similarities. Since natural systems have smooth dynamics, an opportunity is lost if an unsupervised objective function remains static. The absence of concrete supervision suggests that smooth dynamics should be integrated during the training process. Compared to classical static cost functions, dynamic objective functions allow to better make use of the gradual and uncertain knowledge acquired through pseudo-supervision. In this paper, we propose Dynamic Autoencoder (DynAE), a novel model for deep clustering that overcomes a clustering-reconstruction trade-off, by gradually and smoothly eliminating the reconstruction objective function in favor of a construction one. Experimental evaluations on benchmark datasets show that our approach achieves state-of-the-art results compared to the most relevant deep clustering methods. 

## Rechte
Dies ist ein Fork des Github Repositories "nairouz/DynAE". N. Mrabah (2020): Deep Clustering with a Dynamic Autoencoder: From Reconstruction towards Centroids Construction. https://github.com/nairouz/DynAE. 28.05.2021.

## Daten Vorbereitung

Die Daten des FPI Datensatzes sollten sich im Ordner DynAE/ befinden. Hier sollte ein Ordner /images und die styles.csv zu finden sein.

# Durchführung

Conda Environment: 
> conda activate DynAE_Schmedes

Für beide Datensätze steht ein Jupyter Notebook bereit. In diesen ist erklärt welche Schritte durchzuführen sind.

# Citation
  
  ```
  @article{mrabah2019deep,
  title={Deep Clustering with a Dynamic Autoencoder: From Reconstruction towards Centroids Construction},
  author={Mrabah, Nairouz and Khan, Naimul Mefraz and Ksantini, Riadh and Lachiri, Zied},
  journal={arXiv preprint arXiv:1901.07752},
  year={2019}
  }
  
  ```
  
