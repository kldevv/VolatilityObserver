# %%
import function
import config
import pandas as pd
import numpy as np
import re
#%%
# 更新資料
data = config.data()

data_path = data.get_data_path()
base = data.get_input_url()
time_zone = data.get_time_zone()
day_range = data.get_day_range()

print("Update Data")
function.update_data(data_path, base, time_zone, day_range)

# 輸入資料
print("Get Data")
mrkt_dict = function.GetFile(data_path)
strategy_path = data.get_strategy_path()
print("Get Strategy")
stgy_dict = function.read_strategy(strategy_path)

contract_detail = data.get_contract_detail()
open_mrkt = data.get_open_market()
print("Get Contract")
contract_df = function.read_contract(contract_detail, open_mrkt, mrkt_dict)

# 建立參數
if data.get_how_to_cut() == 'Horizontal':
    cut_params = data.get_horizontal_cut()
elif data.get_how_to_cut() == 'Vertical':
    cut_params = data.get_vertical_cut()
params = {
    'How_to_Cut':data.get_how_to_cut(),
    'Cut_Params':cut_params, 
    'Long_Period':data.get_long_period(),
    'Short_Period':data.get_short_period(),
    'Frequency':data.get_frequency(),
    'Roll_Back':data.get_look_back(), 
    'How_to_Choose': data.get_how_to_selecet(),
    'Max_Stgy':data.get_number_open()
    }

print("Get Strategy Onboard")
list_ = function.get_strategy_onboard(mrkt_dict, stgy_dict, open_mrkt, params, contract_df, time_zone)
list_[0] = re.sub("[^0-9]", "", list_[0])[:-3]

ouput_url = data.get_ouput_url()
function.push_result(list_, ouput_url)
print('Done')