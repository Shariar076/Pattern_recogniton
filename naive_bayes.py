#https://chrisalbon.com/machine_learning/naive_bayes/naive_bayes_classifier_from_scratch/
import pandas as pd
import numpy as np


def p_x_given_y(x,mean_y,var_y):
    p=1/np.sqrt(2*np.pi*var_y)*np.exp(-np.square(x-mean_y)/(2*var_y))
    return p

def predict_prob():
    p_setosa = P_setosa * \
p_x_given_y(pl, male_height_mean, male_height_variance) * \
p_x_given_y(pw, male_weight_mean, male_weight_variance) * \
p_x_given_y(sl, male_footsize_mean, male_footsize_variance) * \
p_x_given_y(sw, male_footsize_mean, male_footsize_variance)

p_versicolor = P_female * \
p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance)
p_virginica = P_female * \
p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance)


df =  pd.read_table('iris.csv', sep=',', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
df = df.sample(frac=1).reset_index(drop=True)

df,df2=np.split(df, [100], axis=0)

print(len(df))
array = df2.values
X = array[:,0:4]

# print(len(df1))
# print(df1.head(10))

n_setosa= df['class'][df['class']=='Iris-setosa'].count()
n_versicolor= df['class'][df['class']=='Iris-versicolor'].count()
n_virginica= df['class'][df['class']=='Iris-virginica'].count()

n_total= df['class'].count()

means= df.groupby('class').mean()
# print(means)
vars= df.groupby('class').var()
# print(vars)
setosa_sl_mean= means['sepal-length'][means.index=='Iris-setosa'].values[0]
setosa_sw_mean= means['sepal-width'][means.index=='Iris-setosa'].values[0]
setosa_pl_mean= means['petal-length'][means.index=='Iris-setosa'].values[0]
setosa_pw_mean= means['petal-width'][means.index=='Iris-setosa'].values[0]

setosa_sl_var = vars['sepal-length'][vars.index=='Iris-setosa'].values[0]
setosa_sw_var = vars['sepal-width'][vars.index=='Iris-setosa'].values[0]
setosa_pl_var = vars['petal-length'][vars.index=='Iris-setosa'].values[0]
setosa_pw_var = vars['petal-width'][vars.index=='Iris-setosa'].values[0]



versicolor_sl_mean= means['sepal-length'][means.index=='Iris-versicolor'].values[0]
versicolor_sw_mean= means['sepal-width'][means.index=='Iris-versicolor'].values[0]
versicolor_pl_mean= means['petal-length'][means.index=='Iris-versicolor'].values[0]
versicolor_pw_mean= means['petal-width'][means.index=='Iris-versicolor'].values[0]

versicolor_sl_var= vars['sepal-length'][vars.index=='Iris-versicolor'].values[0]
versicolor_sw_var= vars['sepal-width'][vars.index=='Iris-versicolor'].values[0]
versicolor_pl_var= vars['petal-length'][vars.index=='Iris-versicolor'].values[0]
versicolor_pw_var= vars['petal-width'][vars.index=='Iris-versicolor'].values[0]

virginica_sl_mean= means['sepal-length'][means.index=='Iris-virginica'].values[0]
virginica_sw_mean= means['sepal-width'][means.index=='Iris-virginica'].values[0]
virginica_pl_mean= means['petal-length'][means.index=='Iris-virginica'].values[0]
virginica_pw_mean= means['petal-width'][means.index=='Iris-virginica'].values[0]

virginica_sl_var= vars['sepal-length'][vars.index=='Iris-virginica'].values[0]
virginica_sw_var= vars['sepal-width'][vars.index=='Iris-virginica'].values[0]
virginica_pl_var= vars['petal-length'][vars.index=='Iris-virginica'].values[0]
virginica_pw_var= vars['petal-width'][vars.index=='Iris-virginica'].values[0]


print(virginica_pw_var)