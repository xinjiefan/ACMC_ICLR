import pickle
with open('reinforce/5em5_10_demean/histories_fc.pkl') as a:
    data_rf_demean = pickle.load(a)
cider_rf_demean={}

for i in range(20):
    print(337000 + i*1000)
    print(data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr'])
    if data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr'] is None:
        break
    else:
        cider_rf_demean[337000 + i *1000] = data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr']


with open('reinforce/5em5_10_nodemean/histories_fc.pkl') as a:
with open('histories_fc.pkl') as a:
    data_rf_nodemean = pickle.load(a)
cider_rf_nodemean={}

for i in range(200):
    print(337000 + i*1000)
    print(data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats'])
    if data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats'] is None:
        break
    else:
        cider_rf_nodemean[337000 + i *1000] = data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats']
print(data_rf_nodemean['variance_history'])



for i in range(200):
    print(i*10000 + 10000)

print(data_rf_nodemean['val_result_history'][i*10000 + 10000]['lang_stats'])

with open('binary_tree_coding_2_layer.pkl') as a:
    data_rf_nodemean = pickle.load(a)

data_rf_nodemean['val_result_history'][400000]['lang_stats']


import pickle
with open('histories_att2in.pkl') as a:
    data = pickle.load(a)

for i in data['pseudo_num_length_history']:
    if type(data['pseudo_num_length_history'][i]).__module__ != 'numpy':
      data['pseudo_num_length_history'][i] = data['pseudo_num_length_history'][i].data.cpu().numpy()
from six.moves import cPickle
with open(('histories_att2in.pkl'), 'wb') as f:
    cPickle.dump(data, f)


data['first_order_history'] = data['first_order_history'].data.cpu().numpy()
data['second_order_history'] = data['second_order_history'].data.cpu().numpy()


from six.moves import cPickle
with open(('histories_att2in.pkl'), 'wb') as f:
    cPickle.dump(data, f)


import pickle
import numpy as np

with open('histories_binaryatt2in2.pkl') as a:
    data = pickle.load(a)
for i in sorted(data['val_result_history'].keys()):
    print(i, data['val_result_history'][i]['lang_stats'], data['variance_history'][i])

    pi_list = []
    for j in data['val_result_history'][i]['att_logits_list']:
        pi_list.append(np.mean(1 / (1 + np.exp(-j))))
    print(pi_list)



with open('histories_fc.pkl') as a:
    data = pickle.load(a)
data['val_result_history'][520000]['lang_stats']
data['pseudo_num_history'][520000]



data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)



ars_5_sync_100_10/


import pickle

with open('ars_1_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('ars_1_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)


import pickle

with open('ars_5_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('ars_5_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('ars_10_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('ars_10_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('rf_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('rf_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('sc_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('sc_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('ars_5_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('ars_5_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('mle_sync_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('mle_sync_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)

import pickle

with open('arsm_100_10/histories_fc.pkl') as a:
    data = pickle.load(a)
data['loss_history'] = 0
for i in data['val_result_history']:
    #print(data['val_result_history'][i])
    #print(data['val_result_history'][i]['lang_stats'])
    data['val_result_history'][i]['lang_stats'] = data['val_result_history'][i]['lang_stats'].item()
    data['val_result_history'][i]['loss'] = data['val_result_history'][i]['loss'].item()
from six.moves import cPickle
with open(('arsm_100_10/histories_fc.pkl'), 'wb') as f:
    cPickle.dump(data, f)



import pickle
from six.moves import cPickle
with open('infos_fc-best.pkl') as a:
    data = pickle.load(a)
data['opt']['input_fc_dir'] = '/datadrive//IC/data/cocotalk_fc'


with open(('infos_fc-best.pkl'), 'wb') as f:
    cPickle.dump(data, f)



with open('histories_fc.pkl', 'rb+') as a:
    data = pickle.load(a)



