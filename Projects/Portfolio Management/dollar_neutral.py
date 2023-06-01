import pandas as pd
import numpy as np

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt #brew install python-tk


'''
- "PORTFOLIODIAGNOSTICS" IS A FUNCTION THAT TAKES IN A PORTFOLIO DATAFRAME AND COMPUTES NET EXPOSURES BY SECTORS
- "PNLPERFORMANCE" IS A FUNCTION THAT TAKES IN A PNL SERIES AND COMPUTES SHARPE, RETURN, DRAWDOWN, ETC

- Utility method takes in daily % pnl vector and computes portfolio diagnostics
- Daily % pnl is inputed as a dataframe, with date as index and corresponding pnl for that date as the value

'''

def portfolioDiagnostics(sector_df, signal_df, label):
    #NET SECTOR EXPOSURES
    plt.figure()
    
    d = sector_df.set_index('ticker').to_dict()
    signal_df2 = signal_df.copy(deep=True)
    signal_df2.columns = signal_df2.columns.to_series().map(d['sector'])
    uniquesector = list(set(signal_df2.columns))
    
    for tsector in uniquesector:
        if (str(tsector)=='nan'):
            continue;
            
        tmean = signal_df2.loc[:, tsector].sum(axis=1) #sum up row-wise (across columns for each date) 
        tmean[~np.isfinite(tmean)] = 0
        
        plt.plot(tmean.values, label=tsector)
        plt.legend();
        plt.title(label + " NET EXPOSURE")
        plt.show(block=False)
                
    return


def pnlPerformance(pnl, label):
    cumpnl = pnl.cumsum(skipna = True)

    sharpe = pnl.mean()/np.std(pnl)
    sharpe = sharpe*np.sqrt(252)
    print("")
    print ("PERFORMANCE STATISTICS FOR "+label);
    print("Daily annualized sharpe: "+str(sharpe))
    print ("Average annual returns: "+str(pnl.mean()*252*100)+"%")
    print ("Total returns: " + str(pnl.sum()*100) + "%");
    
    highwatermark_df = cumpnl.cummax();
    drawdown_df = cumpnl - highwatermark_df;
    maxdrawdown = drawdown_df.min();
    print ("Max drawdown: " + str(maxdrawdown*100) + "%");
    
    plt.plot(cumpnl.values, label = label)
    plt.legend()
    plt.show(block=False)
    plt.title("Cumulative PNL chart")
    
    #Compute performance during 'stressed' historical periods
    stressedmarkets = dict()
    stressedmarkets["Covid19"] = (20200301, 20200317);          #Market crash 1H March 2020
    stressedmarkets["Dec18"] = (20181215, 20181231);            #Market crash last 2 weeks
    stressedmarkets["Fall2015"] = (20150701, 20150901);         #Taper tantrum / EU debt crisis.  24 Aug 2015 was "BlackMonday" for Asian, EU and US markets
    stressedmarkets["Oct14"] = (20141001, 20141031);            #Treasury flash crash on 15 Oct 2014
    stressedmarkets["Aug2013"] = (20130820, 20130825);          #Flash freeze on 22 Aug 2013

    for tkey in stressedmarkets.keys():
        mask = pnl.index.to_series().between(stressedmarkets[tkey][0], stressedmarkets[tkey][1])
        print("Stressed period return during " + tkey + ":  " + str(pnl[mask].sum()*100)+"%")
    print("===========================")
    print("")

    
def weight_constraint(signal_df, maxindividualweight = 0.01):
#     maxindividualweight = 0.01  
    for i in range(3): #repeat 3 times to enforce max weight constraint
        signal_df[signal_df > maxindividualweight] = maxindividualweight
        signal_df[signal_df < -maxindividualweight] = -maxindividualweight
        signal_df = signal_df.subtract(signal_df.mean(axis=1), axis='index')
        signal_df = signal_df.divide(signal_df.abs().sum(axis=1), axis='index') #signal_df contains all the normalized demeaned weights
        
    return signal_df


def clean_data(russell_df, universesize=2000):
    vars = ['open', 'high', 'low', 'close', 'volume']
    rawdata = {} #initiate empty dict

    '''
    MAXIMUM FRACTION A SINGLE POSITION CAN TAKE UP OF ENTIRE PORTFOLIO.  
    0.01 MEANS 1%.  i.e. if you have a portfolio of $100 million, 
    max single position size is $1 million
    '''
    maxindividualweight = 0.01 

    #MARKET NEUTRAL STRATEGY BASED ON N DAY REVERSION, BASE UNIVERSE IS STOCKS IN RUSSELL 2000
    for tvar in vars:
        rawdata[tvar] = russell_df.loc[:, ['tickerid', 'ticker', 'date', tvar]]
        rawdata[tvar] = rawdata[tvar].pivot(index = 'date', columns = 'ticker', values = tvar)
        rawdata[tvar] = rawdata[tvar].iloc[:, :universesize] #universe cap

    return rawdata
    

#MEAN REVERSION STRATEGY
def market_neutral(return_df, reversiontimehorizon=10):

    signal_df = -return_df.rolling(reversiontimehorizon, min_periods = 3).mean() #Mean-reversion here
    signal_df = signal_df.subtract(signal_df.mean(axis=1), axis='index') #demean
    signal_df = signal_df.divide(signal_df.abs().sum(axis=1), axis='index')
    signal_df = signal_df.shift(1) #TO AVOID FORWARD BIAS.  WE USE YESTERDAY'S INFORMATION TO EXECUTE AT TODAY'S CLOSE PRICES
       
    #BASED ON YESTERDAY'S INFORMATION, WE EXECUTE AT TODAY'S CLOSE PRICES AND COMPUTE OUR PNL BASED ON TOMORROW'S RETURN
    pnl_df = weight_constraint(signal_df) * return_df.shift(-1) 
    pnl = pnl_df.sum(axis=1) #the sum of columns use axis=1
    pnlPerformance(pnl, "MARKET NEUTRAL")
    marketneutralportfolio = signal_df.copy(deep=True)
    
    return marketneutralportfolio

#SECTOR NEUTRAL STRATEGY BASED ON N DAY REVERSION, ALL STOCKS IN RUSSELL 2000
def sector_neutral(sector_df, signal_df, return_df):
    
    d = sector_df.set_index('ticker').to_dict()
    
    #WE JUST REUSE THE PORTFOLIO WEIGHTS FROM PREVIOUS STRATEGY, SINCE WE ARE JUST NEUTRALIZING DIFFERENTLY HERE
    signal_df2 = signal_df.copy(deep=True) 
    signal_df2.columns = signal_df2.columns.to_series().map(d['sector'])
    uniquesector = list(set(signal_df2.columns))
    
    for tsector in uniquesector:
        if (str(tsector)=='nan'):
            continue;
        tmean = signal_df2.loc[:, tsector].mean(axis=1)
        tmean[~np.isfinite(tmean)] = 0
        
        signal_df2.loc[:, tsector] = signal_df2.loc[:, tsector].subtract(tmean, axis='index');
        
    signal_df = pd.DataFrame(data = signal_df2.values, index = signal_df.index, columns = signal_df.columns)
    signal_df = signal_df.divide(signal_df.abs().sum(axis=1), axis='index')
    
        
    pnl_df = weight_constraint(signal_df) * return_df.shift(-1)
    pnl = pnl_df.sum(axis=1)
    pnlPerformance(pnl, "SECTOR NEUTRAL")
    sectorneutralportfolio = signal_df.copy(deep=True)
    
    return sectorneutralportfolio

#NON MARKET NEUTRAL STRATEGY BASED ON N DAY REVERSION, BASE UNIVERSE IS STOCKS IN RUSSELL 2000
def non_market_neutral(return_df, reversiontimehorizon=10):    

    signal_df = -return_df.rolling(reversiontimehorizon, min_periods = 3).mean()
    signal_df = signal_df.divide(signal_df.abs().sum(axis=1), axis='index') #Never demean
    signal_df = signal_df.shift(1)

    maxindividualweight = 0.01          
    for i in range(3):
        signal_df[signal_df > maxindividualweight] = maxindividualweight
        signal_df[signal_df < -maxindividualweight] = -maxindividualweight
        signal_df = signal_df.divide(signal_df.abs().sum(axis=1), axis='index')
        
    pnl_df = signal_df * return_df.shift(-1)
    pnl = pnl_df.sum(axis=1)
    pnlPerformance(pnl, "NON MARKET NEUTRAL")
    nonmarketneutralportfolio = signal_df.copy(deep=True)
    
    return nonmarketneutralportfolio


def main():
    
#     russell_df = pd.read_csv("data/russell2000pvdata.csv", error_bad_lines = False)
#     sector_df = pd.read_csv("data/sector.csv", error_bad_lines = False)
    
    #Read once
    russell_df = pd.read_csv("../data/russell2000pvdata.csv", on_bad_lines = 'skip')
    sector_df = pd.read_csv("../data/sector.csv", on_bad_lines = 'skip')
    
    raw_data = clean_data(russell_df) #contains ['open', 'high', 'low', 'close', 'volume']
    
    # only using close price
    ret = (raw_data['close'] / raw_data['close'].shift(1)) - 1
    return_df = ret.copy(deep=True)

    marketneutralportfolio = market_neutral(return_df)
    sectorneutralportfolio = sector_neutral(sector_df, marketneutralportfolio, return_df)
    nonmarketneutralportfolio = non_market_neutral(return_df)
    
    
    portfolioDiagnostics(sector_df, marketneutralportfolio, "MARKET NEUTRAL");
    portfolioDiagnostics(sector_df, sectorneutralportfolio, "SECTOR NEUTRAL");
    portfolioDiagnostics(sector_df, nonmarketneutralportfolio, "NON MARKET NEUTRAL");
    
    
    plt.show() #so that plt.figure doesn't close

    
if __name__ == "__main__":
    main()







