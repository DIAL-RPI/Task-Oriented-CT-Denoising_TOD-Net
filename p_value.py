import os
import sys
import numpy as np
from scipy import stats
import pandas as pd

def cal_p_value(a, b):
    #n1 = len(a)
    #n2 = len(b)
    #var_a = a.var(ddof=1)
    #var_b = b.var(ddof=1)

    ## Calculate the t-statistics
    #t = (a.mean() - b.mean())/np.sqrt(var_a/n1+var_b/n2)

    ## Compare with the critical t-value
    #Degrees of freedom
    #df = n1 + n2 - 2

    #p-value after comparison with the t 
    #p = 1 - stats.t.cdf(t,df=df)


    #print("t = " + str(t))
    #print("p = " + str(2*p))
    ### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


    ## Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(a,b)
    #print("t = " + str(t2))
    #print("p = " + str(p2))
    return t2, p2

primary_dir = '/home/xux12/proj/kidseg/models'
metric_names = ['DSC','ASD']
class_names = ['Kidney']

modelpaths = [
    ['{}/model_20200925132323/results_test'.format(primary_dir),'m0'],
    ['{}/model_20201002150852/results_test'.format(primary_dir),'m1'],
    ['{}/model_20201003000029/results_test'.format(primary_dir),'m2'],
    #['{}/model_20201003110034/results_test'.format(primary_dir),'m3'],
    ['{}/model_20201003212837/results_test'.format(primary_dir),'m4']
]


data = {}
failed_case_num = {}
for [modelpath, modelname] in modelpaths:
    df = pd.read_csv('{}/metric_test.csv'.format(modelpath))
    data[modelpath] = df

with open('{}/p-values.txt'.format(sys.path[0]), 'w') as output_file:
    for metric_name in metric_names:
        for cls_id, cls_name in enumerate(class_names):
            cls_label = cls_id + 1
            headline='p-value of metric {} on {}:'.format(metric_name, cls_name)
            print(headline)
            output_file.write(headline+'\n')
            headline = '{0:<16s}'.format('models')
            for [modelpath, modelname] in modelpaths:
                headline += '{0:<16s}'.format(modelname)
            print(headline)
            output_file.write(headline+'\n')
            for [modelpath, modelname] in modelpaths:
                df_a = data[modelpath]
                arr_a = df_a[df_a['Class'] == cls_label][metric_name].to_numpy()
                #print('a mean:{}'.format(arr_a.mean()))
                line = '{0:<16s}'.format(modelname)
                for [compare_model, compare_modelname] in modelpaths:
                    #if compare_model == main_model:
                    #    continue
                    df_b = data[compare_model]
                    arr_b = df_b[df_b['Class'] == cls_label][metric_name].to_numpy()
                    #print('b mean:{}'.format(arr_b.mean()))
                    _, p = cal_p_value(arr_a, arr_b)
                    line += '{0:<16.4f}'.format(p)
                    #print(p)
                print(line)
                output_file.write(line+'\n')
            print('\n')
            output_file.write('\n')