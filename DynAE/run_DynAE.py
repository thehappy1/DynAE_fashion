import sys
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from DynAE import DynAE
from datasets import load_data
import metrics

#config
dataset ='fpi'
loss_weight_lambda=0.5
save_dir='results'
visualisation_dir='visualisation'
data_path ='data' + dataset
batch_size=256
maxiter_pretraining=130e3
maxiter_clustering=1e5
tol=0.01
optimizer1=SGD(0.001, 0.9)
optimizer2=tf.optimizers.Adam(0.0001)
kappa = 3
ws=0.1
hs=0.1
rot=10
scale=0.
gamma=100

#dataloading and model intilization
x, y = load_data(dataset, data_path)
n_clusters = len(np.unique(y))
model = DynAE(batch_size=batch_size, dataset=dataset, dims=[x.shape[-1], 500, 500, 2000, 10], loss_weight=loss_weight_lambda, gamma=gamma, n_clusters=n_clusters, visualisation_dir=visualisation_dir, ws=ws, hs=hs, rot=rot, scale=scale)
model.compile_dynAE(optimizer=optimizer1)
model.compile_disc(optimizer=optimizer2)
model.compile_aci_ae(optimizer=optimizer2)

#Load the pretraining weights if you have already pretrained your network
model.ae.load_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
#model.critic.load_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')

#Pretraining phase
#model.train_aci_ae(x, y, maxiter=maxiter_pretraining, batch_size=batch_size, validate_interval=2800, save_interval=2800, save_dir=save_dir, verbose=1, aug_train=True)


#Save the pretraining weights if you do not want to pretrain your model again
#model.ae.save_weights(save_dir + '/' + dataset + '/pretrain/ae_weights.h5')
#model.critic.save_weights(save_dir + '/' + dataset + '/pretrain/critic_weights.h5')


#Clustering phase
y_pred = model.train_dynAE(x=x, y=y, kappa=kappa, n_clusters=n_clusters, maxiter=maxiter_clustering, batch_size=batch_size, tol=tol, validate_interval=140, show_interval=None, save_interval=2800, save_dir=save_dir, aug_train=True)

#Save the clustering weights
model.ae.save_weights(save_dir + '/' + dataset + '/cluster/ae_weights.h5')

#Validation
model.validate(x, y)
