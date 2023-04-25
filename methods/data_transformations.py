import numpy as np
from scipy.spatial.distance import pdist

def normalised_euclidean_distance(ground_truth, posterior):
    euc_dist = 1-pdist(np.stack((ground_truth, posterior)))[0] / np.linalg.norm(abs(np.array(ground_truth)) + 2*np.ones((1, 6)))
    return euc_dist


def read_array_str(array_str):
    return np.array([float(i) for i in array_str[1:-1].strip().split(' ') if len(i) > 0])


def distance(array1_str, array2_str):
    array1 = read_array_str(array1_str)
    array2 = read_array_str(array2_str)

    return normalised_euclidean_distance(array1, array2)


def concat_name(sign, trial_name):
    if sign in ['pos', 'neg', '1', '2', '3']:
        return trial_name + '_' + sign
    else:
        return trial_name
    

def transform_generic(graph_str, ground_truth, graph_name, transform_indirect):
    graph = read_array_str(graph_str)
    if graph_name.split('_')[0] in ['chain', 'dampened', 'confound']:
        #print(graph_name, read_array_str(ground_truth), '->', read_array_str(ground_truth) * transform_indirect[graph_name])
        return graph * transform_indirect[graph_name]
    else:
        return graph


def tag_column(array_str, tag_dict):
    return tag_dict[array_str]


def add_dampened_tag(df):
    confound_tags = {graph:str(i+1) for i, graph in enumerate(np.sort(df[df.trial_name == 'dampened'].ground_truth.unique()))}
    idx = df[df.trial_name == 'dampened'].index
    df_confound = df[df.trial_name == 'dampened']
    df.loc[idx, 'sign'] = df_confound.apply(lambda x: tag_column(x.ground_truth, confound_tags), axis=1).to_list()
    return df


def split_model_hue(model, internal_state):
    if internal_state == 'mean_field_vis':
        model_type = 'Variational'
    else:
        model_type = 'Standard'
    
    if 'local_computations' in model or 'LC' in model:
        model_factorisation = 'LC'
    else:
        model_factorisation = 'Normative'

    return model_type, model_factorisation