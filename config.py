# %%
'''This is the configuration file for the model
'''
import datetime
class data():
    def __init__(self):
        '''以下為路徑與數值部分======================================================================
        '''
        # 載入資料庫URL 與 策略路徑
        self.input_url = "http://10.229.17.110:8080/cgi-bin/q1min.py?symbol="
        self.data_path = "歷史資料(GMT+0)"
        self.strategy_path = "策略"

        # 日盤區間
        self.day_range = {
                    "Summer":{
                        "TXF1" : (datetime.time(0, 45), datetime.time(5, 45)), 
                        "HSI1" : (datetime.time(1, 15), datetime.time(8, 30)),
                        "SCN1" : (datetime.time(1, 0), datetime.time(8, 36)),
                        "SSI1" : (datetime.time(23, 30), datetime.time(6, 31)),
                        "NQ1" : (datetime.time(12, 30), datetime.time(19, 15)),
                        "YM1" : (datetime.time(12, 30), datetime.time(19, 15)),
                    },
                    "Winter":{
                        "TXF1" : (datetime.time(0, 45), datetime.time(5, 45)), 
                        "HSI1" : (datetime.time(1, 15), datetime.time(8, 30)),
                        "SCN1" : (datetime.time(1, 0), datetime.time(8, 36)),
                        "SSI1" : (datetime.time(23, 30), datetime.time(6, 31)),
                        "NQ1" : (datetime.time(13, 30), datetime.time(20, 15)),
                        "YM1" : (datetime.time(13, 30), datetime.time(20, 15)),
                    }
                }

        # 商品時區
        self.time_zone = {
                    "TXF1" : ('Etc/GMT-8', 8, 8), 
                    "HSI1" : ('Etc/GMT-8', 8, 8),
                    "SCN1" : ('Etc/GMT-8', 8, 8),
                    "SSI1" : ('Etc/GMT-8', 8, 8),
                    "NQ1" : ('US/Eastern', -4, -5),
                    "YM1" : ('US/Eastern', -4, -5)
                }

        # 輸出資料庫URL
        self.ouput_url = {
            'host':'10.229.17.189',
            'user':'aiteam', 
            'password':'aiteam',
            'port':3306, 
            'db':'tradingroom', 
            'table_name':'sys_strategyinfo'
         }

        # 合約細節
        # KEY = 市場代號
        # VALUE = [單口合約大小, 交易貨幣換算台幣倍數]
        self.open_market = ['YM1', 'NQ1', 'SCN1', 'HSI1', 'SSI1', 'TXF1']
        self.contract_detail = {
            'DAX1':[5, 35],
            'YM1':[.5, 30],
            'NQ1':[2, 30],
            'SCN1':[2.5, 30],
            'HSI1':[10, 3.8],
            'SSI1':[100, .28],
            'EXF1':[4000, 1],
            'TXF1':[50, 1],
        }

        '''以下為參數部分======================================================================
        '''
        # 波動度計算參數
        self.frequency = '15T'        # 日內波動計算頻率
        self.short_period = 1        # 短波計算天數 (日內平均)
        self.long_period = 10        # 長波計算天數 (日k平均)

        # 象限切割參數
        self.how_to_cut = 'Horizontal' # or 'Vertical'
        # Vertical_Cut = [短波百分比, 長波百分比, 比較天數]
        self.vertical_cut = [.5, .5, 60]
        # Horizontal_Cut = [標準化天數]
        self.horizontal_cut = [60]

        # 策略選擇參數
        self.how_to_select = 'Expected'        # 如何比較同象限策略績效 # or 'Mean' or 'Sortino' or 'Profitable'
        self.look_back = 1000        # 比較過去 n 次的交易紀錄
        self.number_open = 15        # 最高開啟策略
    

    def get_input_url(self):
        return self.input_url

    def get_data_path(self):
        return self.data_path

    def get_strategy_path(self):
        return self.strategy_path
    
    def get_day_range(self):
        return self.day_range

    def get_time_zone(self):
        return self.time_zone

    def get_ouput_url(self):
        return self.ouput_url
    
    def get_open_market(self):
        return self.open_market
    
    def get_contract_detail(self):
        return self.contract_detail
    
    def get_frequency(self):
        return self.frequency
    
    def get_short_period(self):
        return self.short_period
    
    def get_long_period(self):
        return self.long_period

    def get_how_to_cut(self):
        return self.how_to_cut
    
    def get_vertical_cut(self):
        return self.vertical_cut
    
    def get_horizontal_cut(self):
        return self.horizontal_cut
    
    def get_how_to_selecet(self):
        return self.how_to_select
    
    def get_look_back(self):
        return self.look_back
    
    def get_number_open(self):
        return self.number_open