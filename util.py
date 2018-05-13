import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_bar_counts(df, var, fontsize=16, x=24, y=10):
    df1 = df[[var, 'Income']]
    total = df1[var].value_counts()

    over50 = df1.loc[df1['Income'] == 1]
    x1 = over50.groupby(var).size()
    under50 = df1.loc[df1['Income'] == 0]
    y1 = under50.groupby(var).size()

    ind = df1[var].unique()
    ind.sort()

    #Set general plot properties
    sns.set_style("white")
    sns.set_context({"figure.figsize": (x, y)})

    #Plot 1 - background - "total" (top) series
    sns.barplot(x = ind, y = x1.add(y1, fill_value=0), color = "red")

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x = ind, y=y1, color = "#0000A3")

    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Income <= 50k', 'Income > 50k'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)

    #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Count")
    bottom_plot.set_xlabel(var)
    plt.setp(bottom_plot.get_xticklabels(), rotation=45)

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(fontsize)

def plot_bin_bar_counts(df, var, bins, fontsize=14):
    df1 = df[[var, 'Income']]
    df1[var+'_bin'], edges = pd.cut(df1[var], bins, right=False, retbins=True)
    nice_names = ['{b:0.1f} : {t:0.1f}'.format(b=edges[i], t=edges[i+1]) for i in range(len(edges)-1)]
    df1['x_bin_better'] = pd.cut(df1[var], bins=bins, labels=nice_names)
    df1 = df1.drop(columns=[var])

    over50 = df1.loc[df1['Income'] == 1]
    under50 = df1.loc[df1['Income'] == 0]
    x1 = over50.groupby('x_bin_better').size()
    y1 = under50.groupby('x_bin_better').size()

    df1['total'] = y1 + x1
    z1 = df1.groupby('x_bin_better').size()

    categories = df1['x_bin_better'].cat.categories
    ind = np.array([x for x in categories])

    #Set general plot properties
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})

    #Plot 1 - background - "total" (top) series
    sns.barplot(x = ind, y = x1.add(y1,fill_value=0), color = "red")

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x = ind, y = y1, color = "#0000A3")

    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Income <= 50k', 'Income > 50k'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)

        #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Count")
    bottom_plot.set_xlabel(var)
    plt.setp(bottom_plot.get_xticklabels(), rotation=45)

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(fontsize)

#CITE: 
def native(country):
    if country in ['United-States', 'Cuba']:
        return 'US'
    elif country in ['England', 'Germany', 'Canada', 'Italy', 'France', 'Greece', 'Philippines']:
        return 'Western'
    elif country in ['Mexico', 'Puerto-Rico', 'Honduras', 'Jamaica', 'Columbia', 'Laos', 'Portugal', 'Haiti',
                     'Dominican-Republic', 'El-Salvador', 'Guatemala', 'Peru', 
                     'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Vietnam', 'Holand-Netherlands' ]:
        return 'Poor' # no offence
    elif country in ['India', 'Iran', 'Cambodia', 'Taiwan', 'Japan', 'Yugoslavia', 'China', 'Hong']:
        return 'Eastern'
    elif country in ['South', 'Poland', 'Ireland', 'Hungary', 'Scotland', 'Thailand', 'Ecuador']:
        return 'Poland team'
    else: 
        return country 

def primary(x):
    if x in ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'Primary'
    else:
        return x

# CITE: https://github.com/pandas-dev/pandas/issues/12042
def encode_categorical_data(df, bin_hot=False):
	if bin_hot:
		bin_one_hots = df.loc[:, df.dtypes == np.uint8]
		#Income and sex
		bin_one_hots = bin_one_hots.append(df.loc[:, df.dtypes == np.float64])

	cat_vars = list(df.select_dtypes(include=['object']).columns)

	one_hots = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']]
	
	for var in cat_vars:
		new = pd.get_dummies(df[var], prefix=var)
		one_hots = one_hots.join(new)
		drop_col = var + '_' + df.groupby([var]).size().idxmax()
		one_hots.drop(drop_col, axis=1, inplace=True)
		one_hots.drop(var, axis=1, inplace=True)

	if not bin_hot:
		int_vars = df.loc[:, df.dtypes == np.int64]
	else:
		int_vars = bin_one_hots

	one_hots = one_hots.join(int_vars)

	return one_hots

def encode_bin_data(df):
	cat_vars = list(df.select_dtypes(include=['category']).columns)	

	one_hots = df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'age-hours']]

	for var in cat_vars:
		new = pd.get_dummies(df[var], prefix=var)
		one_hots = one_hots.join(new)
		drop_col = var + '_' + str(df.groupby([var]).size().idxmax())
		one_hots.drop(drop_col, axis=1, inplace=True)
		one_hots.drop(var, axis=1, inplace=True)

	obj_vars = df.select_dtypes(include=['object'])

	one_hots = one_hots.join(obj_vars)
	int_vars = df.loc[:, df.dtypes == np.int64]	
	one_hots = one_hots.append(int_vars)

	return one_hots