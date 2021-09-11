# %%
import config
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import glob
import os
import requests
from io import StringIO



#%%
def GetFile(path):    
    data_dict = {}
    all_data = glob.glob(path + "\\" + "*.csv")

    for filename in all_data:
        key = filename.split(".")[0]
        key = key.split("\\")[1]
        print(key)
        df = pd.read_csv(filename, usecols = [0, 1])
        df.index = pd.to_datetime(df["Date and Time"],  infer_datetime_format=True)
        df = df.drop(["Date and Time",], axis = 1)
        data_dict[key] = pd.DataFrame(df["Close"])
        print(df.index[0], df.index[-1])
    return data_dict

def Is_SummerTime(date, tz, summer_gap, winter_gap):
    delta = (date.tz_localize(None) - date.tz_convert("GMT").tz_localize(None)).total_seconds()//3600
    if delta == summer_gap:
        return True
    elif delta == winter_gap:
        return False
    else:
        raise Exception('Time Calculation Failed')
        return

def FindBreakPoint(date_list, tz, summer_gap, winter_gap):
    BreakPoints = []
    
    if Is_SummerTime(date_list[0], tz, summer_gap, winter_gap) == True:
        BreakPoints.append((date_list[0].tz_convert("GMT").tz_localize(None), "Summer"))
    else:
        BreakPoints.append((date_list[0].tz_convert("GMT").tz_localize(None), "Winter"))
                
    for i in range(1, len(date_list)):
        if Is_SummerTime(date_list[i - 1], tz, summer_gap, winter_gap) == True:
            if Is_SummerTime(date_list[i], tz, summer_gap, winter_gap) == True:
                pass
            else:
                BreakPoints.append((date_list[i].tz_convert("GMT").tz_localize(None), "Winter"))
        else:
            if Is_SummerTime(date_list[i], tz, summer_gap, winter_gap) == True:
                BreakPoints.append((date_list[i].tz_convert("GMT").tz_localize(None), "Summer"))
            else:
                pass
    return BreakPoints

def GetData(base, symbol, start_date, end_date):
    start_str = '{d.year}-{d.month}-{d.day}'.format(d = start_date)
    end_str = '{d.year}-{d.month}-{d.day}'.format(d = end_date)
    
    url = base + symbol + "&BEG=" + start_str + "&END=" + end_str
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception('Web Server Connection Failed')
        
    data = pd.read_csv(StringIO(res.text), header=None, dtype = str)
    data.index = pd.to_datetime(data[0] + data[1])
    data.drop([0,1,2,3,4,6], axis = 1, inplace = True)
    data.columns = ["Close"]
    return data

def update_data(data_path, base, Time_Zone, Day_Range):
    data_dict = GetFile(data_path)

    for key in data_dict.keys():
        df = data_dict[key].copy()
        symbol = key
        print(symbol)
        last_time = df.index[-1]
        start_date = last_time.date()
        end_date = pd.Timestamp.today(tz = 'GMT').date() + datetime.timedelta(days = 1)
        
        data = GetData(base, symbol, start_date, end_date)
        data = data[(data.index > last_time)]
        
        date_list = list(data.index.tz_localize("GMT").tz_convert(Time_Zone[key][0]))
        if (len(date_list) != 0):
            break_point = FindBreakPoint(date_list, Time_Zone[key][0], Time_Zone[key][1], Time_Zone[key][2])
        else:
            break_point = []
        data_list = [df]
        for i in range(len(break_point)):
            if i == (len(break_point) - 1):
                day_range = Day_Range[break_point[-1][1]][key]
                range_diff = day_range[1] > day_range[0]
                
                temp = data[data.index >= break_point[-1][0]].copy()
                cond = (temp.index > last_time)
                if range_diff:
                    cond = cond & (temp.index.time >= day_range[0]) & (temp.index.time <= day_range[1])
                else:
                    cond = cond & ((temp.index.time >= day_range[0]) | (temp.index.time <= day_range[1]))
                temp = temp[cond]
                data_list.append(temp)
            else:
                day_range = Day_Range[break_point[i][1]][key]
                range_diff = day_range[1] > day_range[0]
                
                temp = data[(data.index >= break_point[i][0]) & (data.index < break_point[i + 1][0])].copy()
                cond = (temp.index > last_time)
                if range_diff:
                    cond = cond & (temp.index.time >= day_range[0]) & (temp.index.time <= day_range[1])
                else:
                    cond = cond & ((temp.index.time >= day_range[0]) | (temp.index.time <= day_range[1]))
                temp = temp[cond]
                data_list.append(temp)
            
        result = pd.concat(data_list)
        print(result.index[0], result.index[-1])
        result.index.name = "Date and Time"
        result.to_csv( data_path + "\\" + symbol + ".csv")



def read_strategy(dir):
    count = 0
    strategy_dict = dict()
    arr = os.listdir(dir)[:]
    for file in arr:
        print(file)
        arr_c = os.listdir(dir + "\\" + file)
        sheets = dict()
        for xls in arr_c:
            if xls[:2] != "~$":
                strategy = pd.read_excel(dir + "\\" + file + "\\" + xls, index_col = 0, header = 2, sheet_name = "交易明細").dropna(how = 'all')
                strategy.columns = ["委託單編號","類型","訊號","日期","時間","價格","數量","獲利(¤)","獲利(%)","累積獲利 (¤)","累積獲利 (%)","最大可能獲利(¤)","最大可能獲利 (%)","最大可能虧損 (¤)","最大可能虧損 (%)"]
                strategy['Date and Time'] = pd.to_datetime(pd.to_datetime(strategy['日期']).dt.strftime('%Y/%m/%d') + ' ' + strategy['時間'].astype(str))
                strategy.index = pd.to_datetime(strategy['Date and Time'])
                strategy.drop(["日期"], axis = 1, inplace = True)
                sheets[xls] = strategy
                count = count + 1
        strategy_dict[file] = sheets
    print('read_strategy_succeed')
    return strategy_dict


def read_contract(contract_detail, open_market, market_dict):
    exchange = {}
    for mrkt, value in contract_detail.items():
        if mrkt in open_market:
            mrkt_value = market_dict[mrkt].iloc[-1].values[0]
            contract = value[0]
            currency = value[1]
            down = contract * currency * mrkt_value * 0.01
            exchange[mrkt] = [contract, currency, mrkt_value, down]
    # 一個市場掉1%掉多少
    exchange_df = pd.DataFrame.from_dict(exchange, orient='index')
    exchange_df.columns = ['單口契約點數', '新台幣/使用貨幣', '目前大盤點數', '單口數 1%跳動新台幣']
    # 最大的基數
    exchange_df_base = exchange_df['單口數 1%跳動新台幣'].max()
    # # 所以要如何分配口
    exchange_df['平衡倍數'] = exchange_df_base / exchange_df['單口數 1%跳動新台幣']
    # exchange_df_mul.apply(lambda x : np.round(x))
    exchange_df['最小口數'] = exchange_df['平衡倍數'].apply(lambda x : np.round(x)).values
    exchange_df['口數平衡後  1%跳動新台幣'] = exchange_df['最小口數'] * exchange_df['單口數 1%跳動新台幣']
    return exchange_df



def get_strategy_onboard(mrkt_dict, stgy_dict, open_mrkt, params, contract_df, time_zone):
    how = params['How_to_Cut']
    cut_params = params['Cut_Params']
    criteria = params['How_to_Choose']
    max_stgy = params['Max_Stgy']
    VS_dict = {}
    for key, price in mrkt_dict.items():
        long_vol_period = params['Long_Period']
        intra_vol_time_interval = params['Frequency']
        short_vol_period = params['Short_Period']
        VS = VolatilitySplit(price, key, long_vol_period, intra_vol_time_interval, short_vol_period, time_zone)
        VS.Compute_Volatility()
        if how == 'Vertical':
            VS.set_vertical_parameter(*cut_params)
            VS.Split_Vertical()
        if how == 'Horizontal':
            VS.set_horizontal_parameter(*cut_params)
            VS.Split_Horizontal()
        VS_dict[key] = VS

    Best_Dimension = {}
    Best_Dimension_Value = {}
    Right_Dim = {}
    Contract_Amount = {}
    for mrkt_symbol in open_mrkt:
        #print(mrkt_symbol)
        VS = VS_dict[mrkt_symbol]
        stgy = stgy_dict[mrkt_symbol]
        cluster = VS.GetRecentCluster(how)
        #print(mrkt_symbol, cluster)

        for stgy_name, stgy_sheet in stgy.items():
            #print(stgy_name)
            if how == 'Horizontal':
                df = VS.Back_Test_Horizontal(stgy_sheet, roll_back=params['Roll_Back'])
            else:
                df = VS.Back_Test_Vertical(stgy_sheet, roll_back=params['Roll_Back'])

            if criteria == 'Expected':
                df["Expected"] = df["% Profitable"] * df["% Mean"]
                Best_Dimension[stgy_name] = df["Expected"].idxmax()
                Best_Dimension_Value[stgy_name] = df["Expected"].max()
            elif criteria == 'Profitable':
                Best_Dimension[stgy_name] = df["% Profitable"].idxmax()
                Best_Dimension_Value[stgy_name] = df["% Profitable"].max()
            elif critiera == 'Sortino':
                Best_Dimension[stgy_name] = df["Sortino Ratio"].idxmax()
                Best_Dimension_Value[stgy_name] = df["Sortino Ratio"].max()
            elif critera == 'Mean':
                Best_Dimension[stgy_name] = df["% Mean"].idxmax()
                Best_Dimension_Value[stgy_name] = df["% Mean"].max()

            Same_Dim = (Best_Dimension[stgy_name] == cluster)
            Right_Dim[stgy_name] = Same_Dim

            Contract_Amount[stgy_name] = contract_df.loc[mrkt_symbol, '最小口數']

    Contract_Amount_df = pd.DataFrame.from_dict(Contract_Amount, orient='index')
    Right_Dim_df = pd.DataFrame.from_dict(Right_Dim, orient='index')
    pd.set_option('display.max_rows', 1000)
    #print(Right_Dim_df)
    Best_Dimension_Value_df = pd.DataFrame.from_dict(Best_Dimension_Value, orient='index')
    Best_Dimension_Value_df_filtered_by_Right_Dim= (Right_Dim_df*Best_Dimension_Value_df).replace(0, np.nan)
    Go_Signal = (Best_Dimension_Value_df_filtered_by_Right_Dim.rank(axis=0, ascending=False, method='first', numeric_only=True) <= max_stgy).astype(int)
    #print(Go_Signal)
    
    Final_Amount = Go_Signal * Contract_Amount_df        # Var Final_Amount可以用來檢視結果

    # 轉換成Database需要的Format然後輸出
    output_list = []
    output_dict = {}
    for stgy_name, num in Final_Amount.iterrows():
        name = stgy_name.split(' ')[2]
        output_dict[name] = int(num)
    output_list.append(datetime.datetime.now().isoformat())
    output_list.append('benchmark')
    output_list.append(str(output_dict).translate({ord(i): None for i in "'{} "}))
    
    #print(output_list)
    return output_list


def push_result(aList, database):
    # 建立連線
    import mysql.connector
    mydb = mysql.connector.connect(
    host= database['host'],
    user= database['user'],
    password= database['password'],
    port = database['port'],
    db = database['db']
    )
    mycursor = mydb.cursor()
    print(mydb)

    # 確認今天寫入的資料
    print('本日寫入資料:')
    for x in aList:
        print(x)

    
    # 寫入資料
    sql = "INSERT INTO sys_strategyinfo (Time, Module, StrategyInfo) VALUES (%s, %s, %s)"
    value = tuple(aList)
    mycursor.execute(sql, value)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")


    # 確認DB中的最後五筆資料
    mycursor.execute("SELECT * FROM sys_strategyinfo")
    myresult = mycursor.fetchall()
    print('資料庫中最後五筆:')
    for x in myresult[-5:]:
        print(x)
    print('****成功****')










class VolatilitySplit():
    # 紀錄參數
    def __init__(self, close, symbol, long_vol_period, intra_vol_time_interval, short_vol_period, time_zone, vol_type = "historical"):
        temp = close["Close"].copy()
        temp.index = temp.index.tz_localize("GMT").tz_convert(time_zone[symbol][0]).tz_localize(None)
        self.close = temp
        self.symbol = symbol
        self.long_vol_period = long_vol_period
        self.intra_vol_time_interval = intra_vol_time_interval
        self.short_vol_period = short_vol_period
        self.vol_type = vol_type
        self.time_zone = time_zone

     # 紀錄直向切割參數
    def set_vertical_parameter(self, short_quantile, long_quantile, rolling_interval):
        self.vertical_quantile = short_quantile
        self.vertical_quantile = long_quantile
        self.vertical_rolling_interval = rolling_interval
    # 紀錄橫向切割參數       
    def set_horizontal_parameter(self, rolling_interval):
        self.horizontal_rolling_interval = rolling_interval

    # 計算波動度
    def Compute_Volatility(self):
        # 日內波動度計算 (短天期)
        # 公式: 日內預設頻率 每K棒報酬的標準差 
        intra_price = self.close.resample(self.intra_vol_time_interval).last().dropna()
        intra_ret =  np.log(intra_price / intra_price.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        intra_vol = intra_ret.resample('d').std().replace(0, np.nan).dropna()
        short_vol = intra_vol.rolling(self.short_vol_period).mean().dropna()
        
        # 跨日波動度計算 (長天期)
        daily_price = self.close.resample('d').last().dropna()
        daily_ret = np.log(daily_price / daily_price.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        long_vol = daily_ret.rolling(self.long_vol_period).std().replace(0, np.nan).dropna()
        
        # 選擇波動度的計算方法，預設為歷史波動度
        if self.vol_type == "historical":
            short_vol_s = short_vol.shift(1)
            long_vol_s = long_vol.shift(1)
            # 未來波動度的部分待未來開發:)
        elif self.vol_type == "future":
            short_vol_s = short_vol.shift(1 - short_vol_period)
            long_vol_s = long_vol.shift(1 - long_vol_period)
        # 轉存格式
        self.data = pd.DataFrame([daily_price, short_vol, long_vol]).T.dropna()
        self.data_shift = pd.DataFrame([daily_price, short_vol_s, long_vol_s]).T.dropna()
        self.data.columns = ["Close", "Short_Vol", "Long_Vol"]
        self.data_shift.columns = ["Close", "Short_Vol", "Long_Vol"]
    
    # ==================
    # 計算象限
    # 垂直十字切法
    def Split_Vertical(self):
        threshold = self.vertical_quantile
        window = self.vertical_rolling_interval
        
        # 判斷過去 n 日的波動度百分數
        df = self.data_shift.copy()
        df['Long_threshold']= df["Long_Vol"].rolling(window = window).quantile(threshold)
        df['Short_threshold']= df["Short_Vol"].rolling(window = window).quantile(threshold)
        df = df.dropna()
        
        # 比較本日與過去 n 日的波動度百分數的大小
        cluster_list = []
        for i in df.index:
            x = df.loc[i]
            if (x["Long_Vol"] >= x['Long_threshold']) and (x["Short_Vol"] >= x['Short_threshold']):
                cluster_list.append(0)
            elif (x["Long_Vol"] >= x['Long_threshold']) and (x["Short_Vol"] < x['Short_threshold']):
                cluster_list.append(1)
            elif (x["Long_Vol"] < x['Long_threshold']) and (x["Short_Vol"] < x['Short_threshold']):
                cluster_list.append(2)
            elif (x["Long_Vol"] < x['Long_threshold']) and (x["Short_Vol"] >= x['Short_threshold']):
                cluster_list.append(3)
        # 紀錄結果
        df["Cluster"] = cluster_list
        self.vertical_cluster = df

     # Kmeans切法 (可試試看其他機器學習的分法，目前覺得Kmeans表現最好)       
    def Split_Horizontal(self):
        # 先對資料做移動標準化
        window = self.horizontal_rolling_interval
        df = self.data_shift.copy()
        mean = df.rolling(window).mean()
        std = df.rolling(window).std()
        df = ((df - mean)/std).dropna()
        X = df[['Long_Vol', 'Short_Vol']].values
        
        # 利用Kmeans去分
        kmeans = KMeans(n_clusters = 4, n_init = 100, max_iter = 30000)
        Y = kmeans.fit_predict(X)

        # 紀錄結果
        df["Cluster"] = Y
        self.horizontal_cluster = df
    
    # 垂直十字象限切法，輸入策略，回傳各象限表現
    def Back_Test_Vertical(self, strategy_df, roll_back = 200):
        strategy = strategy_df.dropna()
        df_ver = self.vertical_cluster.reindex(strategy.index, method = "ffill").dropna()
        strategy = strategy[df_ver.index[0]:][["獲利(%)"]]
        strategy["Ver_Cluster"] = df_ver["Cluster"]
        length = min(len(strategy), roll_back)
        strategy = strategy.tail(length)
        
       
        strategy["Clipped"] = strategy['獲利(%)'].clip(upper=0)
        strategy_win = strategy[strategy['獲利(%)'] > 0]
        strategy_lose = strategy[strategy['獲利(%)'] < 0]
        
        temp = strategy[["獲利(%)", 'Ver_Cluster']]
        max_ = temp.groupby('Ver_Cluster').max()
        min_ = temp.groupby('Ver_Cluster').min()
        mean = temp.groupby('Ver_Cluster').mean()
        std = strategy[["Clipped", 'Ver_Cluster']].groupby('Ver_Cluster').std()
        sortino = mean["獲利(%)"] / std["Clipped"]
        
        count = temp.groupby('Ver_Cluster').count()
        win = strategy_win[["獲利(%)", 'Ver_Cluster']].groupby('Ver_Cluster').count()
        lose = strategy_lose[["獲利(%)", 'Ver_Cluster']].groupby('Ver_Cluster').count()
        wpct = win / count
        
        ind_list = [max_, min_, mean, sortino, count, win, lose, wpct]
        output = pd.concat(ind_list, axis=1)
        output.columns = ['% Max', '% Min', '% Mean', 'Sortino Ratio', 'Count', 'Winning Trades', 'Losing Trades', '% Profitable']
        output.fillna(value=0, inplace=True)
        return output

     # Kmeans切法，輸入策略，回傳各象限表現   
    def Back_Test_Horizontal(self, strategy_df, roll_back = 1000):
        strategy = strategy_df.dropna()
        df_hor = self.horizontal_cluster[strategy.index[0]:strategy.index[-1]].reindex(strategy.index, method = "ffill").dropna()
        strategy = strategy[df_hor.index[0]:][["獲利(%)"]]
        strategy["Hor_Cluster"] = df_hor["Cluster"]
        length = min(len(strategy), roll_back)
        strategy = strategy.tail(length)
       
        strategy["Clipped"] = strategy['獲利(%)'].clip(upper=0)
        strategy_win = strategy[strategy['獲利(%)'] > 0]
        strategy_lose = strategy[strategy['獲利(%)'] < 0]
        
        temp = strategy[["獲利(%)", 'Hor_Cluster']]
        max_ = temp.groupby('Hor_Cluster').max()
        min_ = temp.groupby('Hor_Cluster').min()
        mean = temp.groupby('Hor_Cluster').mean()
        std = strategy[["Clipped", 'Hor_Cluster']].groupby('Hor_Cluster').std()
        sortino = mean["獲利(%)"] / std["Clipped"]
        
        count = temp.groupby('Hor_Cluster').count()
        win = strategy_win[["獲利(%)", 'Hor_Cluster']].groupby('Hor_Cluster').count()
        lose = strategy_lose[["獲利(%)", 'Hor_Cluster']].groupby('Hor_Cluster').count()
        wpct = win / count
        
        ind_list = [max_, min_, mean, sortino, count, win, lose, wpct]
        output = pd.concat(ind_list, axis=1)
        output.columns = ['% Max', '% Min', '% Mean', 'Sortino Ratio', 'Count', 'Winning Trades', 'Losing Trades', '% Profitable']
        output.fillna(value=0, inplace=True)
        return output

    # 回傳今天的象限    
    def GetRecentCluster(self, how):
        if how == 'Horizontal':
            return self.horizontal_cluster["Cluster"][-1]
        elif how == 'Vertical':
            return self.vertical_cluster["Cluster"][-1]