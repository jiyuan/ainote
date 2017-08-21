# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 08:02:23 2017

@author: jiyuan

Exploratory Data Analysis
"""

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#import matplotlib as mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sb
sb.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})

## Parameter settings
data_dir = "D:/fintech/model/collection/data/"
ydata_filename = '8b_ydata.data'
xdata_filename = '8b_fact_jm_apply.data'

y_df = pd.read_csv(os.path.join(data_dir, ydata_filename))
print(y_df.head())
print(y_df.info())




#pt = y_df.pivot_table(
#        index='yflag', 
#        columns='marr_status', 
#        values='cus_num', 
#        aggfunc=np.size
#        )
#print(pt.head())


#xdata_filelist = [
#        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
#        if ydata_filename not in f and f.endswith('.data')
#        ]
#print(xdata_filelist)


################################################################################
## Feature Generating

class FeatureGenerater(object):
    prefix = 'fg_'
    
    def __init__(self, y_df, x_df):
        self.x_df = x_df
        self.features = x_df.copy()
    
    def gen(self):
        return self.features

    ## the followings are the lib function for feature generating.
 
    def _aggregating(self):
        #x_int = self.xfile.select_dtypes(include=['int'])
        pass
    
    def _pivoting(self):
        pass
    
    ## Numerical Features
    def gen_num_features(self):
        #x_int = self.xfile.select_dtypes(include=['int'])
        pass
    
    ## Categorical Features
    def gen_cat_features(self):
        pass
    
    ## Temporal Features
    def gen_tem_features(self):
        pass
    
    ## Spatial Features
    def gen_spa_features(self):
        pass


class Df1FG(FeatureGenerater):
    def gen(self):
        pass

#ydf = pd.read_csv('yfile.csv')
#df1 = pd.read_csv('file1.csv')
#df1_gen = Df1FG(ydf, df1).gen()



class OptimizeDF(object):
    
    def __init__(self, df):
        self.df = df
        self.optimized_df = df.copy()
        
    def mem_usage(self, pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else: # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)

    def opt_num_features(self):
        df_int = self.df.select_dtypes(include=['int'])
        converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
        self.optimized_df[converted_int.columns] = converted_int

        df_float = self.df.select_dtypes(include=['float'])
        converted_float = df_float.apply(pd.to_numeric, downcast='float')
        self.optimized_df[converted_float.columns] = converted_float

    def opt_cat_features(self):
        df_obj = self.df.select_dtypes(include=['object']).copy()
        
        converted_obj = pd.DataFrame()
        for col in df_obj.columns:
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if num_unique_values / num_total_values < 0.5:
                converted_obj.loc[:, col] = df_obj[col].astype('category')
            else:
                converted_obj.loc[:, col] = df_obj[col]
        self.optimized_df[converted_obj.columns] = converted_obj

    def opt_tem_features(self):
        df_date = self.optimized_df.date  # .date here is the varname with date format
        self.optimized_df['date'] = pd.to_datetime(df_date, format='%Y%m%d')




################################################################################
## Exploratory Data Analysis

class EDA(object):
    
    def __init__(self, y_df, x_df, targetvar, keyvar):
        self.y_df = y_df
        self.x_df = x_df
        self.target = targetvar
        self.key = keyvar

    def univariate(self, plot=False):
        #n = len(x_float.columns) + len(x_cat.columns)
        print('='*60,"\nExecute univariate analysis.")

        ## judge the duplication
        N = self.x_df[self.key].count()
        n = self.x_df.groupby(self.key).size().count()
        if N > n:
            print("Dataset has duplication, there are %s entries, and %s \
non-duplicated keys." % (N, n))
            x_df_gb = self.x_df.groupby(self.key)
            print('-'*60), print(x_df_gb.size())
            print('-'*60), print(x_df_gb.size().describe())
            
            x_df_gb.agg({'': np.sum,
                         '': np.mean,
                         '': lambda x: np.std(x, ddof=1)})
        else:
            print("No duplication, total entries number is %s" % N)
            print('-'*60)

            ## Numerical features
            x_int = self.x_df.select_dtypes(include=['int'])
            for icol in x_int.columns:
                print('-'*60), print(x_int[icol].info())
                if plot:
                    sb.distplot(x_int[icol], kde=False)
                    sb.plt.show()
    
            x_float = self.x_df.select_dtypes(include=['float'])
            for icol in x_float.columns:
                print('-'*60), print(x_float[icol].info())
                if plot:
                    sb.distplot(x_float[icol], kde=False)
                    sb.plt.show()
        
            ## Categorical features
            x_cat = self.x_df.select_dtypes(include=['category'])
            for icol in x_cat.columns:
                print('-'*60), print(x_cat[icol].value_counts())
                if plot:
                    col_bar = x_cat[icol].value_counts()
                    sb.barplot(x=col_bar.index, y=col_bar.values)
                    sb.plt.show()
    
            ## Temporal features
            x_time = self.x_df.select_dtypes(include=['datetime'])
            for icol in x_time.columns:
                print('-'*60), print(x_time[icol].value_counts())
                if plot:
                    sb.distplot(x_time[icol], kde=False)
                    sb.plt.show()
    
    def bivariate(self, plot=False):
        print("enter bivariate")
        ## 通过targetvar从y_df中提取目标变量，根据keyvar关联x_df和y_df，做二变量分
        ## 析, by boxplot
    
        g = sb.FacetGrid(y_df, col='marr_status')
        g.map(sb.distplot, 'yflag')
        sb.plt.show()
        
        ## combine x_df and y_df
#        df = 
    
#        ## Numerical features
#        x_int = self.x_df.select_dtypes(include=['int'])
#        for icol in x_int.columns:
#            print('='*60)
#            if plot:
#                g = sb.FacetGrid(df, col=icol)
#                 g.map(sb.distplot, self.target)
#                sb.plt.show()
#
#        x_float = self.x_df.select_dtypes(include=['float'])
#        for icol in x_float.columns:
#            print('='*60)
#            print(x_float[col].info())
#            if plot:
#                sb.distplot(x_float[col], kde=False)
#                sb.plt.show()
#    
#        ## Categorical features
#        x_cat = self.x_df.select_dtypes(include=['category'])
#        for icol in x_cat.columns:
#            print('='*60)
#            print(x_cat[col].value_counts())
#            if plot:
#                col_bar = x_cat[col].value_counts()
#                sb.barplot(x=col_bar.index, y=col_bar.values)
#                sb.plt.show()
#
#        ## Temporal features
#        x_time = self.x_df.select_dtypes(include=['datetime'])
#        for icol in x_time.columns:
#            print('='*60)
#            print(x_time[col].value_counts())
#            if plot:
#                sb.distplot(x_time[col], kde=False)
#                sb.plt.show()
    
    def multivariate(self, plot=False):
        print("enter multivariate")
    
    


## Univariate visualization ##
## summary statistics for each field in the raw dataset.

## Numerical features
#sb.distplot(y_df.yflag, kde=False)
#sb.jointplot(data=y_df, x='wa_total', y='yflag', color='g')

## Temporal features


## Spatial features


#sb.set(style="white", palette="muted", color_codes=True)
#fig, axes = plt.subplots(n,1)

## Categorical features
#marr_bar = y_df.marr_status.value_counts()
#sb.barplot(x=marr_bar.index, y=marr_bar.values)

## Bivariate visualization ##
## summary statistics for assessing the relationship between each variable in 
## the dataset and the target variable of interest.

## FacetGrid
#g = sb.FacetGrid(y_df, col='marr_status')
#g.map(sb.distplot, 'yflag')

## Multivariate visualization ##
## understand interactions between different fields in the data.

## HeatMap
#sb.heatmap(pt, annot=True, fmt=".1f")
#sb.boxplot(data=y_df, x='yflag', y='wa_total')

#sb.plt.show()

################################################################################

#for f in xdata_filelist:
#    df = pd.read_csv(f)
#    # print(df.describe())
#    # print(df.head())
    
## 读入y和x，进行变量分析，获得衍生方法

column_types = {
        'cus_num': 'category',
        'flag_id': 'category',
        'flag_cell': 'category',
        'api_code': 'category',
        }
datecolumn_list = ['apply_date']
#x_df = pd.read_csv(os.path.join(data_dir, xdata_filename))
x_df = pd.read_csv(
        os.path.join(data_dir, xdata_filename),
        dtype=column_types,
        parse_dates=datecolumn_list,
        infer_datetime_format=True
        )
print(x_df.head())
print(x_df.info())

x_eda = EDA(y_df, x_df, 'yflag', 'cus_num')

x_eda.univariate(plot=False)
##print('='*60,"\nExecute bivariate analysis.")
##x_eda.bivariate(plot=False)
#print('='*60,"\nExecute multivariate analysis.")
#x_eda.multivariate(plot=False)
print('='*60)


#print(x_df['cus_num'].value_counts())