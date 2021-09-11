def back_test(mrkt_dict, stgy_dict, params, start, end):
    # 判斷每次需要重新測試象限表現的日期
    try:
        start_date = datetime.date(*[int(x) for x in (start.split('-'))])
        end_date = datetime.date(*[int(x) for x in (end.split('-'))])
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.date()
        report_date = []
        for year in range(start_date.year-1, end_date.year+1):
            report_date += [datetime.date(year, 3, 31),
                    datetime.date(year, 6, 30),
                    datetime.date(year, 9, 30),
                    datetime.date(year, 12, 31)]
        report_date = [r for r in report_date if start_date < r < end_date]
        print('Revaluation Date ok!')
    except:
        print('==Revaluation Date Crush==')


    # 判斷每個報表最早的日期與最晚的日期
    try:
        strategy_time_dict = dict()
        for stgy in stgy_dict.values():
            for stgy_name, stgy_sheet in stgy.items():
                strategy_time_dict[stgy_name] = (stgy_sheet.index[0], stgy_sheet.index[-1])
        return strategy_time_dict
        print('Report Date ok!')
    except:
        print('==Report Date Crush==)


    # 建立不同市場的波動度物件
    try:
        how = params['How_to_Cut']
        cut_params = params['Cut_Params']
        VS_list = []
        for key in mrkt_dict.keys():
            close = market_dict[key]
            long_vol_period = params['Long_Period']
            intra_vol_time_interval = params['Frequency']
            short_vol_period = params['Short_Period']
            VS = VolatilitySplit(close, key, long_vol_period, intra_vol_time_interval, short_vol_period)
            VS.Compute_Volatility()
            if how == 'Vertical_Cut':
                VS.set_vertical_parameter(*cut_params)
                VS.Split_Vertical()
            if how == 'Horizontal_Cut':
                VS.set_horizontal_parameter(*cut_params)
                VS.Split_Horizontal()
            VS_list.append(VS)
            print('Volatility Object ok!')
    except:
        print('==Volatility Object Crush=='))

    # 正式回測
    try:
        Best_Dimension_Seasonal = pd.DataFrame(columns = strategy_time_dict.keys(), index = report_date)
        Best_Value_Seasonal = pd.DataFrame(columns = strategy_time_dict.keys(), index = report_date)
        open_mrkt = params['Open_Market']
        roll_back = params['Roll_Back']
        criteria = params['How_to_Evaluate']
        for date in report_date:
            for VS in VS_list:
                mrkt_symbol = VS.symbol
                if mrkt_symbol in open_mrkt:
                    for strategy in stgy_dict[mrkt_symbol].items():
                        start_time = strategy_time_dict[strategy[0]][0]
                        if start_time.date() <= date:
                            df = VS.Back_Test_Horizontal(strategy[1][:date.isoformat()], roll_bacl=roll_back)
                            if criteria == 'Expected':
                                df["Expected"] = df["% Profitable"] * df["% Mean"]
                                Best_Dimension_Seasonal[strategy[0]].loc[date] = df["Expected"].idxmax()
                                Best_Value_Seasonal[strategy[0]].loc[date] = df["Expected"].max()
                            elif criteria == 'Profitable':
                                Best_Dimension_Seasonal[strategy[0]].loc[date] = df["% Profitable"].idxmax()
                                BBest_Value_Seasonal[strategy[0]].loc[date] = df["% Profitable"].max()
                            elif critiera == 'Sortino':
                                Best_Dimension_Seasonal[strategy[0]].loc[date] = df["Sortino Ratio"].idxmax()
                                Best_Value_Seasonal[strategy[0]].loc[date] = df["Sortino Ratio"].max()
                            elif critera == 'Mean':
                                Best_Dimension_Seasonal[strategy[0]].loc[date] = df["% Mean"].idxmax()
                                Best_Value_Seasonal[strategy[0]].loc[date] = df["% Mean"].max()

        Best_Dimension_Seasonal.index = pd.to_datetime(Best_Dimension_Seasonal.index)
        Best_Value_Seasonal.index = pd.to_datetime(Best_Value_Seasonal.index)
        Best_Dimension_Daily = Best_Dimension_Seasonal.resample("1d").first().fillna(method = "ffill")
        Best_Value_Daily = Best_Value_Seasonal.resample("1d").first().fillna(method = "ffill")

        Right_Dim = pd.DataFrame(columns = strategy_time_dict.keys(), index = report_date)
        max_stgy = params['Max_Strategy_Number']
        for VS in VS_list:
            if VS.symbol in open_mrkt:
                df = VS.horizontal_cluster["Cluster"].resample("1d").first().fillna(method = "ffill")[report_date[0].isoformat():report_date[-1].isoformat()]
                strategy_name = list(strategy_dict[VS.symbol].keys())
                for name in strategy_name:
                    '''Demension signal
                    '''
                    Right_Dim[name] = (Best_Dimension_Daily[name] == df).astype(int)
        Right_Dim_Value = (Right_Dim*Best_Value_Daily).replace(0, np.nan)
        Right_Dim_filtered_by_max = ((Right_Dim_Value.rank(axis=1, ascending=False, method='first', numeric_only=True) <= max_stgy).astype(int))

        print('每日象限正確策略:'
        print(Right_Dim.sum(axis=1)))

        print('每日實際開啟策略(限制最大開啟隻數後):')
        print(Right_Dim_filtered_by_max.sum(axis=1))


        open_strategy = {}
        for date, signal in Right_Dim_filtered_by_max .iterrows():
            open_strategy[date] = signal[signal == 1].index
            open_strategy = pd.DataFrame.from_dict(open_strategy, orient='index')
        print('每日開啟策略報表')
        display(open_strategy)

        market_open_strategy = open_strategy.apply(lambda x: [y.split(' ')[0] if type(y) is str else 'N/A' for y in x], axis=1, result_type='expand')
        unique_market_mark = set(market_open_strategy.values.reshape(1, -1)[0])
        market_count = pd.DataFrame()
        for market_mark in unique_market_mark:
            market_count[market_mark] = (market_open_strategy == market_mark).sum(axis=1)
        print('每日市場比數')
        display(market_count)
        market_count.plot.area(figsize=(16, 8))
        plt.show()
        print('總結市場比例')
        market_count.mean().plot(kind='pie', autopct='%1.2f', figsize=(12, 12))

        # 計算進場當天報酬
        Return_df_entry = pd.DataFrame(columns = strategy_time_dict.keys(), index = report_date)
        Return_df_exit = pd.DataFrame(columns = strategy_time_dict.keys(), index = report_date)
        entry_and_exit_price = {}
        long_or_short_dict = {}
        market_date = {}
        for sheets in stgy_dict.items():
            date = {}
            for st in sheets[1].items():
                print(st[0])
                adjusted_st = st[1]
                profit_entry = adjusted_st[["獲利(%)"]].dropna().copy()
                profit_exit = adjusted_st.shift(1)["獲利(%)"].dropna().copy()
                long_or_short = adjusted_st.dropna()[['類型']]
                enter_price = adjusted_st[::2][['價格']].copy()
                exit_price = adjusted_st[1:][::2][['價格']].copy()
                date[st[0]] = pd.DataFrame([profit_entry.index, profit_exit.index]).transpose()
                profit_entry = profit_entry.resample("d").sum().fillna(0)[report_date[0].isoformat():report_date[-1].isoformat()]
                profit_exit = profit_exit.resample("d").sum().fillna(0)[report_date[0].isoformat():report_date[-1].isoformat()]
                Return_df_entry[st[0]] = profit_entry
                Return_df_exit[st[0]] = profit_exit
                long_or_short_dict[st[0]] = long_or_short
                entry_and_exit_price[st[0]] = [enter_price, exit_price]
            market_date[sheets[0]] = date
    

        lookup = Right_Dim_filtered_by_max
        daily_df = pd.DataFrame(index=Best_Profitable_re.index, columns=Best_Profitable_re.columns)
        stgy_daily_return_dict = {}
        stgy_daily_price_dict = {}

        for market, stgy in market_date.items():
            price = whole_dict[market]
            for stgy_name, stgy_action in stgy.items():
                '''策略A每筆交易每天的損益
                '''
                stgy_daily_returns = pd.DataFrame(index = daily_df.index, columns=range(stgy_action.shape[0]))
                stgy_daily_price = pd.DataFrame(index = daily_df.index, columns=range(stgy_action.shape[0]))
                '''跑每筆交易的進出場時間
                '''
                for num, every_action in stgy_action.iterrows():
                    enter = every_action[0]
                    exit = every_action[1]
                    if (enter >= daily_df.index[0]) & (exit <= daily_df.index[-1]):
                        enter_day = datetime.datetime.strptime(enter.strftime('%Y-%m-%d'), '%Y-%m-%d')
                        '''如果那天有被選到象限
                        '''
                        if lookup[stgy_name].loc[enter_day] == 1:
                            entry_price = entry_and_exit_price[stgy_name][0].iloc[num]
                            exit_price = entry_and_exit_price[stgy_name][1].iloc[num]
                            sign = long_or_short_dict[stgy_name].iloc[num].values
                            try:
                                first = price.loc[enter:exit]
                                first[0] = entry_price
                                first = first.resample('d').first()

                                last = price.loc[enter:exit]
                                last[-1] = exit_price
                                last = last.resample('d').last()
                                if sign == '進入Long':
                                    inside = ((last / first) - 1) * 100
                                elif sign == '進入Short':
                                    inside = ((first / last) - 1) * 100
                            except:
                                print('Error')
                                print(num)
                                print(every_action)
                                if sign == '進入Long':
                                    inside = ((exit_price / entry_price) - 1) * 100
                                elif sign == '進入Short':
                                    inside = ((entry_price / exit_price) - 1) * 100
                            stgy_daily_returns.loc[enter-datetime.timedelta(days=1):exit, num] = inside.fillna(0)
                            stgy_daily_price.loc[enter-datetime.timedelta(days=1):exit, num] = first.astype('str') + '/' + last.astype('str')
                stgy_daily_return_dict[stgy_name] = stgy_daily_returns
                stgy_daily_price_dict[stgy_name]  = stgy_daily_price
                stgy_total_returns = stgy_daily_returns.fillna(0).sum(axis=1)
                daily_df[stgy_name] = stgy_total_returns

                actual_returns = (Return_df_entry.fillna(0)*Right_Dim_filtered_by_max)


                # 跑績效圖圖
                fig, ax = plt.subplots(1, 1, figsize=(25, 12))
                for time in range(20):
                    print(time, end=' ')
                    alist = {}
                    for date, df in Return_df_entry.iterrows():
                        return_day = df.sample(n=10)
                        total_return_day = return_day.mean()
                        alist[date] = total_return_day
                    pd.DataFrame.from_dict(alist, orient='index').cumsum().plot(color='lightgray', ax=ax, alpha=.3)
                sns.set(color_codes=True)

                OR = Return_df_entry.mean(axis = 1)

                TARG = actual_returns.sum(aixs=1)/max_stgy
                OR.cumsum().plot(label = "Origin", ax=ax)
                TARG.cumsum().plot(label = criteria, ax=ax)

                actual_action.plot.area(figsize=(20,8), color=['yellow', 'black'])