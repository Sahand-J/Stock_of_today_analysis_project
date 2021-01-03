#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import pandas_datareader,datetime
import pandas_datareader.data as web
from datetime import date 
import operator
import math
import yfinance as yf
import datetime
import yahoo_fin.stock_info as si


# In[2]:


class StockOfToday:
    
    """
    A class used to represent Stock prices Dataframe
    ...

    Attributes
    ----------
    ticker : str
        a formatted string to print out what the animal says
    start_date : str
        the name of the animal
    end_date : str
        the sound that the animal makes
    shares : int
        the number of legs the animal has (default 4)

    """
    
    #
    def __init__(self, ticker, start_date, end_date,shares=0):
        
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.stock = web.DataReader(self.ticker,'yahoo', self.start_date, self.end_date)
        self.shares = shares
        
    #
    def cumel_dict(self):
        
        """Prints what the animals name is and what sound it makes.

        If the argument `sound` isn't passed in, the default Animal
        sound is used.

        Parameters
        ----------
        sound : str, optional
            The sound the animal makes (default is None)

        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a
            parameter.
        """
        
        self.stock['Returns'] = self.stock['Adj Close'].pct_change(1)
        self.stock['Cumulative Return'] = (1 + self.stock['Returns']).cumprod()
        cumel_value = self.stock['Cumulative Return'].iloc[-1].round(4)
        return {self.ticker:cumel_value}
    
    #
    def return_stock_df(self):
        return self.stock
    
    #returns string of ticker symbol
    def return_stock_ticker(self):
        return self.ticker
    
    #returns string of start date
    def return_stock_start_date(self):
        return self.start_date
    
    #returns string of end date
    def return_stock_end_date(self):
        return self.end_date
    
    def return_num_shares(self):
        return self.shares
    
    #determines how many days of date range to calculate std and mean, this case 10%
    def days_to_calc(self):
        return int(round((self.stock['Adj Close'].count())*.1))
    
    #
    def return_stock_div(self):
        div_df = pd.DataFrame(yf.Ticker(self.ticker).dividends)       
        if(div_df.size==0):
            date_time_str = self.start_date
            date_time_obj = datetime.datetime.strptime(date_time_str, '%m-%d-%Y')
            return pd.DataFrame(columns=['Dividends'],index=[date_time_obj]).fillna(0)
        else:    
            return div_df
    
    #
    def return_last_adj_close(self):
        return self.return_stock_df()['Adj Close'][-1]
    
    #
    def rolling_mean_and_std_df(self):
        delta = self.days_to_calc()
        self.stock['Rolling Mean'] = self.stock['Adj Close'].rolling(delta).mean()
        self.stock['Rolling STD'] = self.stock['Adj Close'].rolling(delta).std()
        return self.stock[['Adj Close','Rolling STD','Rolling Mean']]
    
    #
    def kde_df(self):
        return self.stock['Adj Close'].pct_change(1)
    
    #
    def cumelative_return(self):
        self.stock['Returns'] = self.stock['Adj Close'].pct_change(1)
        self.stock['Cumulative Return'] = (1 + self.stock['Returns']).cumprod()
        return self.stock['Cumulative Return']
    
    #
    def stock_finantial_df(self): 
        try:
            finantial_df = si.get_balance_sheet(self.ticker)
        except KeyError as k:
            print('\n')
            print('Error with '+str(k)+': stock must not have ballance sheet, try non-ETFs for this information')
            print('\n')
            
        total_liabil = finantial_df.loc['totalLiab'] #correct
        total_stock_holder_equity = finantial_df.loc['totalStockholderEquity']
        total_assets = finantial_df.loc['totalAssets'] #correct

        df1 = pd.DataFrame(total_liabil)
        df2 = pd.DataFrame(total_stock_holder_equity)
        df3 = pd.DataFrame(total_assets)

        liab_asset_df = pd.concat([df1,df2,df3],axis=1)

        liab_asset_df['debt_to_equity(D/E)'] = liab_asset_df['totalLiab']/liab_asset_df['totalStockholderEquity']

        #print('<1 more equity than debt')
        #print('>1 more debt than equity')
        #print('Debt to equity ratio (total liability/total amount of shareholder equity) < 0.3 ')
        return liab_asset_df
    

    #revenue returns and testing
    def return_yearly_rev_earning(self):
        try:
            rev = si.get_earnings(self.ticker)
        except KeyError as k:
            print('\n')
            print('Error with '+str(k)+': stock must not have ballance sheet, try non-ETFs for this information')
            print('\n')
        yearly_rec_df = rev.get('yearly_revenue_earnings')
        yearly_rec_df.set_index(yearly_rec_df['date'],inplace=True)
        return yearly_rec_df[['revenue','earnings']]
    
    #
    def return_total_company_revenue(self):
        return pd.DataFrame(si.get_income_statement(self.ticker).loc['totalRevenue'])
        
 
    


# In[ ]:





# In[3]:


class stock_of_today_list():
    
    #
    #sharpe_allo_num=0
    #
    
    #constructor to initialise stock of today list object
    def __init__(self,stock_list,allo):
        self.stock_list = stock_list
        #self.sharpe_allo = sharpe_allo
        self.sharpe_allo_num = allo
    #
    def return_stock_list(self):
        return self.stock_list
    
    #
    def SP500_benchmark_df(self):
        SP500 = web.DataReader('SPY','yahoo',self.stock_list[0].return_stock_start_date(),self.stock_list[0].return_stock_end_date())
        SP500['SP500_KDE'] = SP500['Adj Close'].pct_change(1)
        SP500['Returns'] = SP500['Adj Close'].pct_change(1)
        SP500['Cumulative Return'] = (1 + SP500['Returns']).cumprod()
        return SP500
    
    #
    def colors_helper(self,n): 
        ret = []
        r = np.random.randint(256)
        step = 256 / n 
        for i in range(n): 
            r += step 
            r = int(r) % 256 
            R=round(r/256,3)
            ret.append(R)
        return ret 
    
    #
    def colors(self,n): 
        ret = []
        s =len(self.stock_list)
        R=self.colors_helper(s)
        G=self.colors_helper(s)
        B=self.colors_helper(s)
        for i in range(0,s):
            ret.append((R[i],G[i],B[i]))
        return ret
    
#INDIVIDUAL MENU 2

    #
    def volitility(self):
        color_list = self.colors(len(self.stock_list))
        fig, sub_plots = plt.subplots(figsize=(15,10),squeeze = False)
        ind = 0
        for i in self.stock_list:
            i.kde_df().plot(kind='kde',ax=sub_plots[0,0],label=i.return_stock_ticker(),c=color_list[ind])
            ind+=1
        self.SP500_benchmark_df()['SP500_KDE'].plot(kind='kde',ax=sub_plots[0,0],label='SP500',alpha=.6,ls='--',c='red')
        plt.legend()
        plt.show()


# INDIVIDUAL MENU 3

    #
    def cumelative_return(self):
        dict_append = {}
        #n rows by 1 col
        fig, sub_plots = plt.subplots(len(self.stock_list),1,figsize=(20,10),squeeze=False)
        for stock in range(0,(len(self.stock_list))):
            dict_append.update(self.stock_list[stock].cumel_dict())
            y_axis = (1 + self.stock_list[stock].return_stock_df()['Adj Close'].pct_change(1)).cumprod()
            sub_plots[stock,0].plot(y_axis.index,y_axis)
            sub_plots[stock,0].set_ylabel(self.stock_list[stock].return_stock_ticker())
        sorted_d = sorted(dict_append.items(), key=operator.itemgetter(1))
        print('Sorted cumelative returns: '+ str(sorted_d)  )      
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.style.use('seaborn')
        plt.show()
    
    #sorted dictionary of stock ticker and cumelative return
    def cumel_box_plot(self):
        dict_append = {}
        for i in self.stock_list:
            dict_append.update(i.cumel_dict())
    
        sorted_d = dict(sorted(dict_append.items(), key=operator.itemgetter(1)))
        print(sorted_d)
        df = pd.DataFrame.from_dict(sorted_d,orient='index',columns=['value'])
        y = df['value']
        low = min(y)
        high = max(y)
        
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.bar(df.index,y) 
        
        #horsizontal line
        plt.axhline(y = 1.2, color = 'r', linestyle = '--') 
        plt.axhline(y = 1.4, color = 'blue', linestyle = '--') 
        
        plt.style.use('seaborn')
        plt.show()
    
#INDIVIDUAL MENU 1

    #plots a cumilative return graph of each stock based on the given time frame    
    def mean_std_plot(self):
        for i in self.stock_list:
            i.rolling_mean_and_std_df().plot(figsize=(20,8))
        plt.title(i.return_stock_ticker())
        plt.show()
    #
    def dividend_plot(self):
        print('Dividend data based on Yahoo Finance free data\n') 
        for i in self.stock_list:
            fig, sub_plots = plt.subplots(1,3,figsize=(20,10),squeeze = False)
            dividend_df = i.return_stock_div()
            
            if(dividend_df.size<1):
                print('NO DIVIDEND INFO FOR: '+ i.return_stock_ticker())

            merged_stock_div_df = i.return_stock_df().merge(dividend_df, left_index=True, right_index=True)
        
        #within time frame dividend data
            sub_plots[0, 0].plot(merged_stock_div_df.index,merged_stock_div_df['Dividends'],label=str(i.return_stock_ticker()))
            sub_plots[0, 0].set_xlabel('most recent dividend data')
            sub_plots[0, 0].set_ylabel('$ per Share')

        #historic dividend data
            x = dividend_df.index
            y = dividend_df['Dividends']
            sub_plots[0, 1].plot(x,y,label=str(i.return_stock_ticker()))
            sub_plots[0, 1].plot(x,y.rolling(int(round((dividend_df.size*.15)))).mean()) #rolling mean
            sub_plots[0, 1].set_xlabel('historic dividend data')
           
        #relevant time frame
            x_r = dividend_df.iloc[int(round((dividend_df.size/2))):].index
            y_r = dividend_df.iloc[int(round((dividend_df.size/2))):]['Dividends']
            sub_plots[0, 2].plot(x_r,y_r,label=str(i.return_stock_ticker()))
            sub_plots[0, 2].plot(x_r,y_r.rolling(int(round((dividend_df.size*.1)))).mean(),label='Rolling Mean') #rolling mean
            sub_plots[0, 2].set_xlabel('relevant dividend data')
        
            plt.legend()
            plt.tight_layout()
            plt.show()

        fig.autofmt_xdate()
        
    #use for dividend df to combine with regular stock df    
    def return_div_df(self):
        empty_dict = {}
        col_vals = ['last adj close','div $/share(Q)','div $/Y',
                    'div %/Y','#shares','equity',
                    'div-equity-year','div%per-year']
        
        for i in self.stock_list:
        #index
            ticker = i.return_stock_ticker().upper()
        #0    
            last_adj_close = i.return_last_adj_close().round(3)
        #1 
            price_per_share = i.return_stock_div()['Dividends'][-1]
        #2    
            price_per_year = price_per_share*4
        #3    
            percent_per_year_percent = round((price_per_year/last_adj_close)*100,2)
        #4
            num_shares = i.return_num_shares()
        #5 
            equity = num_shares*last_adj_close
        #6
            div_equi_per_year = (num_shares*price_per_share)*4
        #7
            div_percent_per_year = price_per_year/last_adj_close
        
        # Dataframe columns
            div_features = [last_adj_close, price_per_share, price_per_year, 
                            percent_per_year_percent, num_shares, equity, div_equi_per_year,div_percent_per_year]

            empty_dict.update({ticker:div_features})
        
        return pd.DataFrame.from_dict(empty_dict,columns=col_vals,orient ='index').sort_values(by='div%per-year',ascending=False)
    
    #
    def dividend_info(self):
        fig, sub_plots = plt.subplots(1,2,figsize=(20,10),squeeze = False)
        
        x1 = self.return_div_df()['div%per-year'].index
        y1 = self.return_div_df()['div%per-year']
        sub_plots[0, 0].bar(x1,y1,label='Dividend % return per year per Share')
        sub_plots[0, 0].set_ylabel('div%per-year')
        
        #HORIZ line
        upper_lim = sub_plots[0,0].axhline(0.0166,color='red',ls='--')
        lower_lim = sub_plots[0,0].axhline(0.0444,color='blue',ls='--')
        
        x2 = self.return_div_df()['div $/Y'].index
        y2 = self.return_div_df()['div $/Y']
        
        x3= self.return_div_df()['div $/share(Q)'].index
        y3= self.return_div_df()['div $/share(Q)']
        
        sub_plots[0, 1].bar(x2,y2,label='Dividend return per year per Share ($)')
        sub_plots[0, 1].bar(x3,y3,label='Dividend return per Quarter per Share ($)')
        
        plt.legend()
        plt.tight_layout()
        fig.autofmt_xdate()
        plt.show()
    
    #
    def stock_summary(self):
        for s1 in self.stock_list:
            fig, sub_plots = plt.subplots(2,2,figsize=(15,10),squeeze = False)

        #rolling_mean_std_plot
            x = s1.rolling_mean_and_std_df().index
            y1 = s1.rolling_mean_and_std_df()['Adj Close']
            y2 = s1.rolling_mean_and_std_df()['Rolling Mean']
            y3 = s1.rolling_mean_and_std_df()['Rolling STD']
            
            sub_plots[0,0].plot(x,y1,label='Adj Close')
            sub_plots[0,0].plot(x,y2,label='Rolling Mean')
            sub_plots[0,0].plot(x,y3,label='Rolling STD')
            
            sub_plots[0, 0].set_title('Stock Trends')
            sub_plots[0, 0].set_ylabel('Stock Price')
            sub_plots[0, 0].legend()
            
        #kde plot
            #renamed_kde_df = s1.kde_df().rename(columns = {'kde_y_val':s1.return_stock_ticker()})
            s1.kde_df().plot(kind='kde',ax=sub_plots[0,1],label='KDE: Volitility')
            self.SP500_benchmark_df()['SP500_KDE'].plot(kind='kde',ax=sub_plots[0,1],label='SP500',alpha=.6,ls='--',c='red')
            sub_plots[0,1].set_title('KDE Plot')
            sub_plots[0, 1].legend()
            
        #historic dividend trend
            renamed_div = s1.return_stock_div().rename(columns = {'Dividends':'Historic Dividend Data'})
            renamed_div.plot(ax=sub_plots[1,0])
            
            renamed_div_mean = s1.return_stock_div().rename(columns = {'Dividends':'Dividend Mean'})
            renamed_div_mean.rolling(int(round((s1.return_stock_div().size*.1)))).mean().plot(ax=sub_plots[1,0],label='Dividend Rolling Mean') 
            
            sub_plots[1,0].set_ylabel('Dividend return per share ($)')
            sub_plots[1,0].set_title('Historic Dividend Trends')
            sub_plots[1,0].legend()
        
        #returns vs sp500
            s1.cumelative_return().plot(ax=sub_plots[1,1],label=s1.return_stock_ticker())
            self.SP500_benchmark_df()['Cumulative Return'].plot(ax=sub_plots[1,1],label='SP500',alpha=.6,ls='--',c='red')
            sub_plots[1,1].set_ylabel('Cumelative Return (%)')
            sub_plots[1,1].set_title('Cumelative Returns VS SP500')
            sub_plots[1,1].legend()
            
        plt.legend()
        plt.show()
    
    #
    def return_ticker_list(self):
        temp = []
        for i in self.stock_list:
            temp.append(i.return_stock_ticker())
        return temp
    #
    def stock_list_df(self):
        temp = []
        temp2 = []
        for i in self.stock_list:
            temp.append(i.return_stock_df()['Adj Close'])
            temp2.append(i.return_stock_ticker())
        stock = pd.concat(temp,axis=1)
        stock.columns=temp2
        return stock
    #
    def return_last_price(self):
        temp = []
        for i in self.stock_list:
            temp.append(i.return_stock_df()['Adj Close'][-1])
        return temp
    #
    def sharpe_alloc(self,graph=False):    
        
        num_ports=10000
        np.random.seed(101)
        stocks = self.stock_list_df()
        log_return = np.log(stocks/stocks.shift(1))
        all_weights = np.zeros((num_ports,len(self.return_ticker_list())))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for i in range(num_ports):
            weights = np.array(np.random.random(len(self.return_ticker_list())))
            weights = weights / np.sum(weights)

            all_weights[i,:] = weights
            ret_arr[i] = np.sum(log_return.mean()*weights *252)
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
            sharpe_arr[i] = ret_arr[i] /vol_arr[i]

        max_sharpe= sharpe_arr[i].max()
        
        print('\n')
        print('Sharpe ratio: '+ str(max_sharpe))
        print('\n')
        
        max_sharpe_index_loc = sharpe_arr.argmax()
    
        #Sharpe Calculations
        ser = pd.Series(data=all_weights[max_sharpe_index_loc,:],index=self.return_ticker_list())
        df = pd.DataFrame(ser,columns=['% to Allocate'])
        df['Dollar amount to Allocate($)'] = df['% to Allocate'].apply(lambda x: x*float(self.sharpe_allo_num))
        df.insert(2, 'last_price', self.return_last_price(), True) 
        df['#of shares to buy'] = df['Dollar amount to Allocate($)']/df['last_price']
        df['Round up nearest Share'] = df['#of shares to buy'].apply(lambda x: math.ceil(x))
        
        #if we want graph, display it
        if(graph):
            plt.figure(figsize=(10,20))
            plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Volitility')
            plt.ylabel('Return')
            plt.scatter(vol_arr[max_sharpe_index_loc],ret_arr[max_sharpe_index_loc],c='red')
            plt.show()
            
            return df.sort_values(by=['% to Allocate'],ascending = False)
            
        return df.sort_values(by=['% to Allocate'],ascending = False)
    
    #returns dataframe with dividend information
    def return_div_df(self):
        empty_dict = {}
        col_vals = ['last adj close','div $/share(Q)','div $/Y',
                    'div %/Y','#shares','equity',
                    'div-equity-year','div%per-year']

        for i in self.stock_list:
        #index
            ticker = i.return_stock_ticker().upper()
        #0    
            last_adj_close = i.return_last_adj_close().round(3)
        #1 
            price_per_share = i.return_stock_div()['Dividends'][-1]
        #2    
            price_per_year = price_per_share*4
        #3    
            percent_per_year_percent = round((price_per_year/last_adj_close)*100,2)
        #4
            num_shares = i.return_num_shares()
        #5 
            equity = num_shares*last_adj_close
        #6
            div_equi_per_year = (num_shares*price_per_share)*4
        #7
            div_percent_per_year = price_per_year/last_adj_close

        # Dataframe columns
            div_features = [last_adj_close, price_per_share, price_per_year, 
                            percent_per_year_percent, num_shares, equity, div_equi_per_year,div_percent_per_year]

            empty_dict.update({ticker:div_features})

        return pd.DataFrame.from_dict(empty_dict,columns=col_vals,orient ='index').sort_values(by='div%per-year',ascending=False)

    #returns dataframe with dividend and company information/metrics
    def total_df(self):
        div_df = self.return_div_df()
        sharpe_df = self.sharpe_alloc()
        total_df = pd.concat([div_df,sharpe_df],axis=1)
        temp = []
        for i in self.stock_list:
            temp.append(i.stock_finantial_df()['debt_to_equity(D/E)'][0])
        total_df.insert(1, "debt_to_equity(D/E)",temp, True)
        return total_df
    
    
    #
    def plot_company_rev_trends(self):
        for i in self.stock_list:
            
            Y_AXIS_SCALE = 1000000000
            
            yearly_rec_df = i.return_yearly_rev_earning() / Y_AXIS_SCALE
            net_income_df = i.return_total_company_revenue()

            fig, sub_plots = plt.subplots(1,2,figsize=(15,5),squeeze = False)

            yearly_rec_df['revenue'].plot(ax=sub_plots[0,0],label='Revenue')
            sub_plots[0, 0].set_ylabel('Revenue (x 100 million)')
            sub_plots[0, 0].legend()

            #(net_income_df['totalRevenue']/(Y_AXIS_SCALE)).plot(ax=sub_plots[0,0],label='Total Revenue')
            #sub_plots[0, 0].set_ylabel('Total Revenue (x 100 million)')
            #sub_plots[0, 0].legend()

            yearly_rec_df['earnings'].plot(ax=sub_plots[0,1],label='Earnings/Net Income')
            sub_plots[0, 1].set_ylabel('Earnings (x 100 million)')
            sub_plots[0, 1].legend()


            plt.tight_layout()
            fig.autofmt_xdate()
            plt.show()
    
        


# In[4]:


################################################################################


# In[5]:


def stock_list_menu():
    
    try:
        s = stock_menu_looper(0)
    except:
        print('\n')
        print("Stock Ticker Not Found, Try again!")
        print('\n')
        return main_menu()
        
#LIST MENU        
    if s.stock_list_len()>1:
        while 1>0:
            print('list menu')
            print (30 * "-" , "Stock-List Menu" , 30 * "-")
            print("""
                1. Compare Volitility

                2. Cumelative box plot

                3. Dividend Trends (dont plot empty plot)

                4. Dividend Returns""")

            print (67 * "-")
            ans = input("How do you want to analyse these stock?\n ")

            #1    
            if ans=="1":
                return s.init_stock_obj().volitility()
            #2
            elif ans=="2":
                return s.init_stock_obj().cumel_box_plot()
            #3
            elif ans == "3":
                return s.init_stock_obj().dividend_plot()
            #4
            elif ans == "4":
                return s.init_stock_obj().dividend_info()
            #5
            elif ans =="5":
                return main_menu()
            else:
                break;

#INDIVIDUAL MENU
    else:
        while 1>0:
            print (30 * "-" , "Individual Stock Menu" , 30 * "-")
            print("""

        1. Plot the standard deviation and average trend

        2. Plot volitility 

        3. Plot cumelative return

        4. Show summary

        5. Company Revenue Trends

        6. dividend info

        7. Main menu

                   """)
            print (67 * "-")
            ans = input("How do you want to analyse this stock?\n ")

            #1    
            if ans=="1":
                return s.init_stock_obj().mean_std_plot()
            #2    
            elif ans=="2":
                return s.init_stock_obj().volitility()
            #3
            elif ans=="3":
                return s.init_stock_obj().cumelative_return()
            #4
            elif ans=="4":
                return s.init_stock_obj().stock_summary()
            #5
            elif ans=="5":
                return s.init_stock_obj().plot_company_rev_trends()
            #6
            elif ans =="6":
                return s.init_stock_obj().dividend_plot()
            #7
            elif ans =="7":
                return main_menu()
            
            #back to menu
            else:
                return main_menu()


# In[ ]:





# In[6]:


class stock_menu_looper():
    
    def __init__(self, alloc):
        
        start = input('Start date:')
        end = input('End date: ')
        
        self.alloc = alloc
        
        #empty date
        if(len(start)<1 or len(end)<1):
            print('Need Start Date')
            return main_menu()
        
        list_of_stocks = input('Enter comma sporated list of Stock tickers: ')
        my_stocks = list_of_stocks.split(',')
        my_stock_objects = []
        
        if(self.alloc==0):
            for i in my_stocks:
                temp = StockOfToday(i.upper(),start,end)
                my_stock_objects.append(temp)
            self.s = stock_of_today_list(my_stock_objects,allo=0)
        
        # new SOT with sharpe allocation 
        else:
            for i in my_stocks:
                temp = StockOfToday(i.upper(),start,end)
                my_stock_objects.append(temp)
                
            self.s = stock_of_today_list(my_stock_objects,allo = self.alloc)
    #
    def init_stock_obj(self):
        return self.s
    
    #
    def stock_list_len(self):
        return len(self.s.return_stock_list())
    


# In[7]:


def portfolio_menu():
    
    print('List Menu')
    print (30 * "-" , "Stock-List Menu" , 30 * "-")
    print("""
        1. Sharpe Ratio and Allocation
        2. Useful Info (fix cols repeating)
        3. Main Menu
        """)
    print (67 * "-")
    ans = input("How do you want to analyse these stock?\n ")

    #1    
    if ans=="1":
        allo = input('how much to allocate?')
        to_graph = input('Would you like to see the Graph? (y/n)') 
        
        #
        try:
            s = stock_menu_looper(allo)
        except:
            print('\n')
            print('Stock Ticker Not Found, Try again!2')
            print('\n')
            return portfolio_menu()
        #
        
        if(str(to_graph)=="y"):
            return s.init_stock_obj().sharpe_alloc(graph=True)
        else:
            return s.init_stock_obj().sharpe_alloc()

    #2
    elif ans =="2":
        allo = input('how much to allocate?')
        try:
            s = stock_menu_looper(allo)
            df = s.init_stock_obj().total_df()
            #cols_of_interest = [['last adj close','debt_to_equity(D/E)','div $/Y','div%per-year','% to Allocate']]
            #return df.loc[:, (df != 0).any(axis=0)]
            return df[['last adj close','debt_to_equity(D/E)','div $/Y','div%per-year','% to Allocate']].loc[:, (df != 0).any(axis=0)]
        
        except:
            print('\n')
            print('Stock Ticker Not Found, Try again')
            print('\n')
            return portfolio_menu()
            
        
        #print(df.loc[:, (df != 0).any(axis=0)])
        return df.loc[:, (df != 0).any(axis=0)]

    #3
    elif ans =="3":
        return main_menu()

            


# In[ ]:





# In[8]:


def main_menu():
    print (30 * "-" , "MENU" , 30 * "-")
    print("""
        1. Analyse an individual or list of stock
        2. Dividend Portfolio (IN TESTING)
        3. Quit
       """)
    print (67 * "-")
    
    ans = input("What would you like to do?\n ")
    
    #1 main     
    if ans=="1":
        return stock_list_menu()
    
    #2 portfolio
    elif ans=="2":
        return portfolio_menu() 
    
    #3 exit
    elif ans =="3":
        print("\n Goodbye")
    else:
        print("\n Not a valid entry, try again")
        return main_menu()
    


# In[12]:


main_menu()
