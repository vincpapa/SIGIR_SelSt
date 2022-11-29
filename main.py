from Pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce

if __name__ == '__main__':
    model = pd.read_csv('data/EASER.tsv', sep='\t')
    # model = model.loc[model['Recall'] < 0.06]
    # model = model.loc[model['APLT'] < 0.045]
    # model = pd.read_csv('data/tree_mean_results.tsv', sep='\t')
    population = 18070  # 6306
    utopia_point = np.array([1, 1])  # np.array([1, 0])
    reference_point = np.array([0, 0])  # np.array([-0.5, 9*(10 ** (-5))])
    # model = pd.read_csv('data/nn_mean_results.tsv', sep='\t')
    # model = pd.read_csv('data/rec_EASER_cutoff_10_relthreshold_0_2022_11_22_01_35_36.tsv', sep='\t')
    # model['APLT'] = model['APLT'].map(lambda x: x*10)
    # model = pd.read_csv('data/rec_MultiVAE_cutoff_10_relthreshold_0_2022_11_22_01_14_51.tsv', sep='\t')
    obj = ObjectivesSpace(model, {'Recall': 'max', 'APLT': 'max'})  # {'ndcg@10': 'max', 'time_per_doc': 'min'}
    print('****** OPTIMAL *****')
    print(obj.get_nondominated())
    print('****** DOMINATED *****')
    print(obj.get_dominated())
    obj.plot(obj.get_nondominated(), obj.get_dominated(), reference_point)
    nome_framework = obj.nome_framework([utopia_point] * population, True)  # * 6306 trees nn
    # print(obj.nome_framework([np.array([1, 0])] * 6306))
    nome_framework_v2 = obj.nome_framework_v2([utopia_point] * population)
    # print(obj.nome_framework_v2([np.array([1, 0])]*6306))
    hypervolumes = obj.hypervolumes(reference_point)  # np.array([-0.5, 2*(10 ** (-5))]) nn # np.array([-0.5, 9*(10 ** (-5))]) tree
    # print(obj.hypervolumes(np.array([-0.5, 20*(10 ** (-5))]))) # 9*(10 ** (-5))
    knee_point = obj.knee_point()
    # print(obj.knee_point())
    euclidean_distance = obj.euclidean_distance(utopia_point)  # * 11 trees  # * 5 nn
    # print(obj.euclidean_distance([np.array([1, 0])] * 11))
    weighted_mean = obj.weighted_mean(np.array([0.5, 0.5]))
    # print(obj.weighted_mean(np.array([0.5, 0.5])))
    dfList = [nome_framework[['model', 'nostro_framework']], nome_framework_v2[['model', 'nostro_framework_v2']],
              hypervolumes[['model', 'hypervolume']],
              knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
              weighted_mean[['model', 'weighted_mean']]]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='model'), dfList)
    print(df.to_latex(index=False))

