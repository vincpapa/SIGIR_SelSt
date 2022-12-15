from Pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce

if __name__ == '__main__':
    model = pd.read_csv('data/EASER.tsv', sep='\t')
    # model = model.loc[model['Recall'] < 0.06]
    # model = model.loc[model['APLT'] < 0.045]
    # model = pd.read_csv('data/nn_mean_results.tsv', sep='\t')
    population = 18070  # 18070  # 6306
    utopia_point = np.array([1, 1])  # np.array([1, 0]) # np.array([1, 1])  # np.array([1, 0])
    personalized_up = pd.read_csv('data/goodreads_utopia_point.tsv', sep='\t')
    reference_point = np.array([0, 0])  # np.array([-0.5, 2*(10 ** (-5))]) # np.array([0, 0])  # np.array([-0.5, 9*(10 ** (-5))])
    scale = False  # True # False
    # model = pd.read_csv('data/nn_mean_results.tsv', sep='\t')
    # model = pd.read_csv('data/rec_EASER_cutoff_10_relthreshold_0_2022_11_22_01_35_36.tsv', sep='\t')
    # model['APLT'] = model['APLT'].map(lambda x: x*10)
    # model = pd.read_csv('data/rec_MultiVAE_cutoff_10_relthreshold_0_2022_11_22_01_14_51.tsv', sep='\t')
    obj = ObjectivesSpace(model, {'Recall': 'max', 'APLT': 'max'}) # {'Recall': 'max', 'APLT': 'max'})  # {'ndcg@10': 'max', 'time_per_doc': 'min'}
    print('****** OPTIMAL *****')
    print(obj.get_nondominated())
    print('****** DOMINATED *****')
    print(obj.get_dominated())
    obj.plot(obj.get_nondominated(), obj.get_dominated(), reference_point)
    # nome_framework = obj.nome_framework([utopia_point] * population, scale)  # * 6306 trees nn
    nome_framework = obj.nome_framework(personalized_up, scale)
    # print(obj.nome_framework([np.array([1, 0])] * 6306))
    # nome_framework_v2 = obj.nome_framework_v2([utopia_point] * population)
    # print(obj.nome_framework_v2([np.array([1, 0])]*6306))
    hypervolumes = obj.hypervolumes(reference_point)  # np.array([-0.5, 2*(10 ** (-5))]) nn # np.array([-0.5, 9*(10 ** (-5))]) tree
    # print(obj.hypervolumes(np.array([-0.5, 20*(10 ** (-5))]))) # 9*(10 ** (-5))
    knee_point = obj.knee_point(scale)
    # print(obj.knee_point())
    euclidean_distance = obj.euclidean_distance(utopia_point, scale)  # * 11 trees  # * 5 nn
    # print(obj.euclidean_distance([np.array([1, 0])] * 11))
    weighted_mean = obj.weighted_mean(np.array([0.5, 0.5]), scale)
    # print(obj.weighted_mean(np.array([0.5, 0.5])))
    #dfList = [nome_framework[['model', 'ndcg@10', 'time_per_doc', 'nostro_framework']],
    #          hypervolumes[['model', 'hypervolume']],
    #          knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
    #          weighted_mean[['model', 'weighted_mean']]]
    dfList = [nome_framework[['model', 'Recall', 'APLT', 'nostro_framework']],
             hypervolumes[['model', 'hypervolume']],
             knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
             weighted_mean[['model', 'weighted_mean']]]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='model'), dfList)
    # print(df.to_latex(index=False))
    selected = np.zeros(df.shape[0], dtype=np.int8)
    selected[np.argmin(df.values[:, 3], axis=0)] = 1
    selected[np.argmax(df.values[:, 4], axis=0)] = 2
    selected[np.argmax(df.values[:, 5], axis=0)] = 3
    selected[np.argmin(df.values[:, 6], axis=0)] = 4
    selected[np.argmax(df.values[:, 7], axis=0)] = 5
    df['selected'] = pd.Series(selected)
    #df['selected'] = df['nostro_framework'].map(lambda x: 1 if x==min(df['nostro_framework'])
    # df['selected'] = df['hypervolume'].map(lambda x: 1 if x == max(df['hypervolume']) else )

    print(df)
    # df.to_csv('results/EASER.csv', index=False)
    print(np.argmin(df.values[:, 1:], axis=0))
    print(np.argmax(df.values[:, 1:], axis=0))

