# -*- coding: utf-8 -*-
"""
Plotting libarry helpful in building and assessing ML models.

"""
 #%% import some necessary librairies
import os
os.chdir('.')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
from matplotlib import gridspec
import seaborn as sns
import scipy as sp
import warnings
import statsmodels.api as sm

from sklearn.linear_model import Ridge
from scipy import stats
from scipy.stats import norm, skew, boxcox #for some statistics
from statsmodels.graphics.gofplots import ProbPlot

class Diagnostic():
    """
    Plot diagnostics for linear regression.
    """
    def __init__(self, X, y, lin):
        self.df = X.copy()
        self.df['y'] = y.copy()
        self.model = lin
        self.df['position'] = np.arange(len(self.df))
        leverage = (X * np.linalg.pinv(X).T).sum(1)
        rank = np.linalg.matrix_rank(X)
        ff = X.shape[0] - rank
        # Compute the MSE from the residuals
        fitted = self.model.predict(X).T[0]
        residuals = self.df['y'] - fitted
        self.df['residual'] = residuals
        self.df['fitted'] = fitted
        self.mse = np.dot(residuals, residuals) / ff
        ## Compute Cook's distance
        # residual studentized = residual divided by an estimate of its standard deviation
        residuals_studentized = residuals / np.sqrt(self.mse) / np.sqrt(1 - leverage)
        residuals_studentized_abs_sqrt = np.sqrt(np.abs(residuals_studentized))
        self.df['residual_stud'] = residuals_studentized
        self.df['residual_sqrt'] = residuals_studentized_abs_sqrt
        self.df['leverage'] = leverage
        distance = residuals_studentized ** 2 / X.shape[1]
        distance *= leverage / (1 - leverage)
        self.df['distance'] = distance
        # Compute the p-values of Cook's Distance
        # TODO: honestly this was done because it was only in the statsmodels
        # implementation... I have no idea what this is or why its important.
        self.df['p_values'] = sp.stats.f.sf(distance, X.shape[1], ff)
        # Compute the influence threshold rule of thumb
        self.influence_threshold = 4 / X.shape[0]
        self.outlier_percentage = (sum(distance > self.influence_threshold) / X.shape[0])
        self.outlier_percentage *= 100.0
    
    def plotCook(self, annot=0, ax=None):
        """
        Draws a stem plot where each stem is the Cook's Distance of the instance at the
        index specified by the x axis. Optionaly draws a threshold line.
        """
        
        if not ax:
            ax = plt.subplot()
        # Draw a stem plot with the influence for each instance
        _, _, baseline = ax.stem(self.df['distance'], linefmt="C0-", markerfmt=",",
            use_line_collection=True)
        # No padding on either side of the instance index
        ax.set_xlim(0, len(self.df['distance']))
        # Draw the threshold for most influential points
        label = r"{:0.2f}% > $I_t$ ($I_t=\frac {{4}} {{n}}$)".format(self.outlier_percentage)
        ax.axhline(self.influence_threshold, ls="--", label=label, c=baseline.get_color(),
            lw=baseline.get_linewidth(),)
        # Set the title and axis labels
        ax.set_title("Cook's Distance")
        ax.set_xlabel("instance index")
        ax.set_ylabel("influence (I)")
        ax.legend(loc="best", frameon=True)
        if annot:
            dist_sort = np.flip(np.argsort(np.abs(self.df['distance'].values)), 0)
            dist_sort_top = dist_sort[:annot]
            dist_sort_top = self.df['distance'].index[dist_sort_top]
            for r, i in enumerate(dist_sort_top):
                #ax.annotate(i, xy=(i, self.df['distance'][i]))
                ax.annotate(i, xy=(self.df['position'][i], self.df['distance'][i]))
        return ax
    
    def plotResFit(self, annot=0, ax=None):
        #sns.lmplot(x='fitted', y='residual', data=self.df, = 'y',
        #           lowess=True, line_kws={'lw': 2, 'color': 'orange'})
        # plot it
        if not ax:
            fig = plt.figure(figsize=(8, 6)) 
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax = []
            ax.append(plt.subplot(gs[0]))
            ax.append(plt.subplot(gs[1]))
        sns.scatterplot(x='fitted', y='residual', data=self.df, ax=ax[0])
        sns.regplot(x='fitted', y='residual', data=self.df, scatter=False, ax=ax[0], lowess=True,
                    line_kws={'lw': 2, 'color': 'purple'})
        sns.histplot(data=self.df, y='residual', bins=20, kde=True, ax=ax[1])
        ax[0].set_title('Residual vs. Fitted')
        ax[0].set_xlabel('Fitted values')
        ax[0].set_ylabel('Residuals')
        ax[1].set_ylabel('')
        if annot:
            res_sort = np.flip(np.argsort(np.abs(self.df['residual'].values)), 0)
            res_sort_top = res_sort[:annot]
            res_sort_top = self.df['residual'].index[res_sort_top]
            for r, i in enumerate(res_sort_top):
                ax[0].annotate(i, xy=(self.df['fitted'][i], self.df['residual'][i]))
        return ax

    def plotResSqrtFit(self, annot=0, ax=None):
        # plot it
        if not ax:
            ax = plt.subplot()
        sns.scatterplot(x='fitted', y='residual_sqrt', data=self.df, ax=ax)
        sns.regplot(x='fitted', y='residual_sqrt', data=self.df, scatter=False, ax=ax, lowess=True,
                    line_kws={'lw': 2, 'color': 'purple'})
        ax.set_title('Scale-Location')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('$\sqrt{|Standardized Residuals|}$')
        if annot:
            res_sqrt_sort = np.flip(np.argsort(np.abs(self.df['residual_sqrt'].values)), 0)
            res_sqrt_sort_top = res_sqrt_sort[:annot]
            res_sqrt_sort_top = self.df['residual_sqrt'].index[res_sqrt_sort_top]
            for r, i in enumerate(res_sqrt_sort_top):
                ax.annotate(i, xy=(self.df['fitted'][i], self.df['residual_sqrt'][i]))
        return ax
        
    def plotQQ(self, annot=0, ax=None):
        if not ax:
            ax = plt.subplot()
        QQ = ProbPlot(self.df['residual_stud'])
        QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax)
        ax.set_title('Normal Q-Q')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals');
        # annotations
        if annot:
            abs_norm_resid = np.flip(np.argsort(np.abs(self.df['residual_stud'])), 0)
            abs_norm_resid_top = abs_norm_resid[:annot]
            abs_norm_resid_top = self.df['residual_stud'].index[abs_norm_resid_top]
            for r, i in enumerate(abs_norm_resid_top):
                ax.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                               self.df['residual_stud'][i]))
        return ax
    
    def plotResLev(self, annot=0, ax=None):
        if not ax:
            ax = plt.subplot()
        sns.scatterplot(x='leverage', y='residual_stud', data=self.df, ax=ax)
        sns.regplot(x='leverage', y='residual_stud', data=self.df, scatter=False, ax=ax, lowess=True,
                    line_kws={'lw': 2, 'color': 'purple'})
        ax.set_xlim(0, max(self.df['leverage'].values)+0.01)
        ax.set_ylim(-3, 5)
        ax.set_title('Residuals vs Leverage')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals');
        
        #annotations
        if annot:
            leverage_top = np.flip(np.argsort(self.df['leverage'].values), 0)[:annot]
            leverage_top = self.df['leverage'].index[leverage_top]
            for i in leverage_top:
                ax.annotate(i, xy=(self.df['leverage'][i],
                                               self.df['residual_stud'][i]))
        return ax
    
    def plotDiagnostic(self, annot=0):
        fig = plt.figure(figsize=(20, 15), facecolor='w', edgecolor='k')
        gs = fig.add_gridspec(3, 11)
        #Residual vs. fitted
        ax1 = fig.add_subplot(gs[0, :5])
        ax2 = fig.add_subplot(gs[0, 5])
        self.plotResFit(annot=annot, ax=[ax1, ax2])
        #QQ-plot
        ax3 = fig.add_subplot(gs[0, 6:])
        self.plotQQ(annot=annot, ax=ax3)        
        #Residual stad sqrt vs. fitted
        ax4 = fig.add_subplot(gs[1, :5])
        self.plotResSqrtFit(annot=annot, ax=ax4)          
        #Residual  vs. leverage
        ax5 = fig.add_subplot(gs[1, 6:])
        self.plotResLev(annot=annot, ax=ax5)
        #Cooks plot
        ax6 = fig.add_subplot(gs[2, :])
        self.plotCook(annot=annot, ax=ax6)
        gs.tight_layout(fig)
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]
        
    def removeMax(self, type='residual', number=0, return_df = False):
        if type=='residual' or type=='QQ':
            res_sort = np.flip(np.argsort(np.abs(self.df['residual'].values)), 0)
            res_sort_top = res_sort[:number]
            res_sort_top = self.df['residual'].index[res_sort_top]
            print("Deleting entries with index: ", res_sort_top)
            self.df.drop(res_sort_top, inplace=True)
        elif type == 'cook':
            dist_sort = np.flip(np.argsort(np.abs(self.df['distance'].values)), 0)
            dist_sort_top = dist_sort[:number]
            dist_sort_top = self.df['distance'].index[dist_sort_top]
            print("Deleting entries with index: ", dist_sort_top)
            self.df.drop(dist_sort_top, inplace=True)
        elif type == 'leverage':
            leverage_top = np.flip(np.argsort(self.df['leverage'].values), 0)[:number]
            leverage_top = self.df['leverage'].index[leverage_top]
            print("Deleting entries with index: ", leverage_top)
            self.df.drop(leverage_top, inplace=True)
        else:
            print("Warning: wrong plot type chosen.")
            print("Repeat the command with a proper 'type'.")
        if return_df:
            df_ret = self.df.drop(['y', 'residual', 'fitted', 'residual_stud', 'residual_sqrt',
                          'leverage', 'distance', 'p_values', 'position'], axis=1).reset_index(drop=True)
            return df_ret
        
    def returnData(self):
            df_ret = self.df.drop(['residual', 'fitted', 'residual_stud', 'residual_sqrt',
                  'leverage', 'distance', 'p_values', 'position'], axis=1).reset_index(drop=True)
            X = df_ret.loc[:, df_ret.columns != 'y']
            y = df_ret.loc[:, df_ret.columns == 'y']
            return X, y
