# 4th-ML100Days

### 機器學習概論 Introduction of Machine Learning
* **Day_01 : 資料介紹與評估指標**
    * 探索流程 : 找到問題 -> 初探 -> 改進 -> 分享 -> 練習 -> 實戰
    * 思考關鍵點 :
        * 為什麼這個問題重要
        * 資料從何而來
        * 資料型態是什麼
        * 回答問題的關鍵指標是什麼
* **Day_02 : 機器學習概論**
    * 機器學習範疇 : 深度學習(Deep Learning) ⊂ 機器學習(Machine Learning) ⊂ 人工智慧 (Artificial Intelligence)
    * 機器學習是什麼 :
        * 讓機器從資料找尋規律與趨勢不需要給定特殊規則
        * 給定目標函數與訓練資料，學習出能讓目標函數最佳的模型參數
    * 機器學習總類 :
        * 監督是學習(Supervised Learning) : 圖像分類(Classification)、詐騙偵測(Fraud detection)，需成對資料(x,y)
        * 非監督是學習(Unsupervised Learning) : 降維(Dimension Reduction)、分群(Clustering)、壓縮，只需資料(x)
        * 強化學習(Reinforcement Learning) : 下圍棋、打電玩，透過代理機器人(Agent)與環境(Environment)互動，學習如何獲取最高獎勵(Reward)，例如 Alpha GO
* **Day_03 : 機器學習流程與步驟**
    * 資料蒐集、前處理
        * 政府公開資料、Kaggle 資料
            * 結構化資料 : Excel 檔、CSV 檔
            * 非結構化資料 : 圖片、影音、文字
        * 使用 Python 套件
            * 開啟圖片 : `PIL`、`skimage`、`open-cv`
            * 開啟文件 : `pandas`
        * 資料前處理 :
            * 缺失值填補
            * 離群值處理
            * 標準化
    * 定義目標與評估準則
        * 回歸問題？分類問題？
        * 預測目標是什麼？(target or y)
        * 用什麼資料進行預測？(predictor or x)
        * 將資料分為 :
            * 訓練集，training set
            * 驗證集，validation set
            * 測試集，test set
        * 評估指標
            * 回歸問題(預測值為實數)
                * RMSE : Root Mean Squeare Error
                * MAE : Mean Absolute Error
                * R-Square
            * 分類問題(預測值為類別)
                * Accuracy
                * [F1-score](https://en.wikipedia.org/wiki/F1_score)
                * [AUC](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF)，Area Under Curve
    * 建立模型與調整參數
        * Regression，回歸模型
        * Tree-base model，樹模型
        * Neural network，神經網路
        * Hyperparameter，根據對模型了解和訓練情形進行調整
    * 導入
        * 建立資料蒐集、前處理(Preprocessing)等流程
        * 送進模型進行預測
        * 輸出預測結果
        * 視專案需求調整前後端
* **Day_04 : 讀取資料與分析流程(EDA，Exploratory Data Analysis)**   
    * 透過視覺化和統計工具進行分析
        * 了解資料 : 獲取資料包含的資訊、結構、特點
        * 發現 outlier 或異常數值 : 檢查資料是否有誤
        * 分析各變數間的關聯性 : 找出重要的變數
    * 收集資料 -> 數據清理 -> 特徵萃取 -> 資料視覺化 -> 建立模型 -> 驗證模型 -> 決策應用
### 資料清理與數據前處理 Data Cleaning and Preprocessing
* **Day_05 : 如何建立一個 DataFrame？如何讀取其他資料？**
    * 用 [pd.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) 來建立，[練習網站](https://github.com/guipsamora/pandas_exercises)
    * CSV
        ```py
        import pandas as pd
        df = pd.read_csv('example.csv') # sep=','
        df = pd.read_table('example.csv') # sep='\t'
        ```
    * 文本(txt)
        ```py
        with open('example.txt','r') as f:
            data = f.readlines()
        print(data)
        ```
    * Json
        ```py
        import json
        with open('example.json','r') as f:
            data = json.load(f)
        print(data)
        ```
    * 矩陣檔(mat)
        ```py
        import scipy.io as sio
        data = sio.load('example.mat')
        ```
    * 圖像檔(PNG/JPG...)
        ```py
        import cv2
        image = cv2.imread('example.jpg') # Cv2 會以 GBR 讀入
        image = cv2.cvtcolor(image,cv2.COLOR_BGR2RGB)
        ```
    * Python npy
        ```py
        import numpy as np
        arr = np.load('example.npy')
        ```    
    * Picke(pkl)
        ```py
        import pickle
        with open('example.pkl','rb') as f:
            arr = pickle.load(f)
        ```
* **Day_06 : 欄位的資料類型介紹及處理**
    * 資料類型 :
        * 離散變數 : 房間數量、性別、國家
        * 連續變數 : 身高、花費時間、車速
        * 常見的[資料類型](https://blog.csdn.net/claroja/article/details/72622375) : 
            * float64 : 浮點數，可表示離散或連續變數
            * int64 : 整數，可表示離散或連續變數
            * object : 包含字串，表示類別型變數
                * Label encoding : 對象有序資料，例如年齡
                    ```py
                    from sklearn.preprocessing import LabelEncoder
                    df[col] = LabelEncoder().fit_transform(df[col])
                    ```
                * One Hot encoding : 對象無序資料，例如國家
                    ```py
                    df = pd.get_dummies(df)
                    ```
            * 其他 : 日期、boolean
* **Day_07 : 特徵類型**
    * 數值型特徵 : 有不同轉換方式，函數/條件式都可以
    * 類別型特徵 : 通常一種類別對應一種分數
    * 二元型特徵 : True/False，可當類別或數值處理
    * 排序型特徵 : 有大小關係時當數值型特徵處理
    * 時間型特徵 : 可當數值或類別處理，但會失去週期性，需特別處理
* **Day_08 : EDA之資料分佈**
    * 以單變量進行分析
        * 計算集中趨勢
            * 平均數，`mean()`
            * 中位數，`median()`
            * 眾數，`mode()`
        * 計算資料分散程度
            * 最小值，`min()`
            * 最大值，`max()`
            * 範圍
            * 四分位差，`quantile()`
            * 變異數，`var()`
            * 標準差，`std()`
    * 視覺化方式
        * [matplotlib](https://matplotlib.org/gallery/index.html)
            ```py
            import matplotlib.pyplot as plt
            %matplotlib inline # 內嵌方式，不需要 plt.show()
            %matplotlib notebook # 互動方式
            plt.style.use('ggplot') # 改變繪圖風格
            plt.style.available # 查詢有哪些風格
            ```
        * [seaborn](https://seaborn.pydata.org/examples/index.html)
            ```py
            import seaborn as sns
            sns.set()   # 使用預設風格
            sns.set_style('whitegrid')  # 變更其他風格
            sns.axes_style(...) # 更詳細的設定
            ```
    * 延伸閱讀 : [敘述統計與機率分佈](http://www.hmwu.idv.tw/web/R_AI_M/AI-M1-hmwu_R_Stat&Prob_v2.pdf)
* **Day_09 : 離群值(Outlier)及其處理**
    * 異常值出現可能原因
        * 未知值隨意填補或約定成俗代入
        * 錯誤紀錄/手誤/系統性錯誤
    * 檢查異常值的方法
        * 統計值 : 如平均數、標準差、中位數、分位數
            ```py
            df.describe()
            df[col].value_counts()
            ```
        * 畫圖 : 如值方圖、合圖、次數累積分佈等
            ```py
            df.plot.hist()  # 直方圖
            df.boxplot()    # 盒圖
            df = df[col].value_counts().sort_index().cumsum()
            plt.plot(list(df.index), df/df.max())   # 次數累積圖
            ```
    * 異常值的處理方法
        * 取代補植 : 中位數、平均數等
        * 另建欄位 : 標示異常(Y/N)
        * 整欄不用
    * 延伸閱讀 : [辨識異常值](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)、[IQR](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
* **Day_10 : 數值特徵-去除離群值**
    * 方法一 : 去除離群值，可能刪除掉重要資訊，但不刪除會造
    成特徵縮放(標準化/最大最小化)有很大問題
        ```py
        mask = df[col] > threshold_lower & df[col] < threshold_upper
        df = df[mask]
        ```
    * 方法二 : 調整離群值
        ```py
        df[col] = df[col].clip(lower, upper) # 將數值限制在範圍內
        ```
* **Day_11 : 數值填補與連續數值標準化**
    * 常用於填補的統計值
        * 中位數(median) : `np.median(df[col])`
        * 分位數(quantiles) : `np.quantile(df[col],q=...)`
        * 眾數(mode) : 
            ```py
            from scipy.stats import mode
            mode(df[col])
            ```
        * 平均數(mean) : `np.mean(df[col])`
    * 連續型數值[標準化](https://blog.csdn.net/pipisorry/article/details/52247379)
        * 為何要標準化 : 每個變數 x 對 y 的影響力不同
        * 是否要做標準化 : 對於權重敏感或損失函數平滑有幫助者
            * 非樹狀模型 : 線性回歸、羅吉斯回歸、類神經，對預測會有影響
            * 樹狀模型 : 決策樹、隨機森林樹、梯度提升樹，對預測不會有影響
        * 優點 : 加速模型收斂，提升模型精準度
        * 缺點 : 量的單位有影響力時不適用
        * 標準化方法 :
            * Z-transform: $ \frac{(x - mean(x))}{std(x)} $
            * Range (0 ~ 1): $ \frac{x - min(x)}{max(x) - min(x)} $
            * Range (-1 ~ 1): $ (\frac{x - min(x)}{max(x) - min(x)} - 0.5) * 2 $
* **Day_12 : 數值型特徵 - 補缺失值與標準化**
    * [缺失值處理](https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c) : 最重要的是欄位領域知識與欄位中的非缺數值，須注意不要破壞資料分佈
        * 填補統計值 :
            * 填補平均數(mean) : 數值型欄位，偏態不明顯
            * 填補中位數(meadian) : 數值型欄位，偏態明顯
            * 填補重數(mode) : 類別型欄位
        * 填補指定值 : 須對欄位領域知識已有了解
            * 補 0 : 空缺原本就有零的含意
            * 補不可能出現的數值 : 類別型欄位，但不是合用眾數時
            ```py
            df.fillna(0)
            df.fillna(-1)
            df.fillna(df.mean())
            ```
        * 填補預測值 : 速度慢但精確，從其他欄位學得填補知識
            * 若填補範圍廣，且是重要的特徵欄位時可用本方式
            * 須提防 overfitting : 可能退化成其他特徵組合
        * 標準化 : 以合理的方式平衡特徵間的影響力
            * 標準化(Standard Scaler) : 假設數值為常態分
            佈，適用本方式平衡特徵，不易受極端值影響
            * 最大最小化(MinMax Scaler) : 假設數值為均勻分布，適用本方式平衡特徵，易受極端值影響
                ```py
                from sklearn.preprocessing import MinMaxScaler, StandardScaler
                df_temp = MinMaxScaler().fit_transform(df)
                df_temp = StandardScaler().fit_transform(df)
                ```
* **Day_13 : 常用的 DataFrame 操作
    * [Panda 官方 Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
    * [Panda Cheet Sheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
    * 轉換與合併
        * 將"column"轉換成"row" 
            ```py
            df.melt(id_vars=['A'], value_vars=['B', 'C'])
            ```
        * 將"row"轉換成"column"
            ```py
            df.pivot(index='A', columns='B', values='C')
            ```
        * 沿"row"進行合併
            ```py
            pd.concat([df1,df2])
            ```
        * 沿"column"進行合併
            ```py
            pd.concat([df1,df2],axis=1)
        * 以"id"欄位做全合併(遺失以na補)
            ```py
            pd.merge(df1,df2,on='id',how='outer')
            ```
        * 以'id'欄位做部分合併
            ```py
            pd.merge(df1,df2,on='id',how='inner')
            ```
    * subset 操作
        * 邏輯操作(>,<,=,&,|,~,^)
            ```py
            sub_df = df[df.age>20]
            ```
        * 移除重複的"row"
            ```py
            df = df.drop_duplicates()
            ```
        * 前 5 筆、後 n 筆
            ```py
            sub_df = df.head()   # default = 5
            sub_df = df.tail(10)
            ```
        * 隨機抽樣
            ```
            sub_df = df.sample(frac=0.5)    # 抽50%
            sub_df = df.sample(n=10)    # 抽10筆
            ```
        * 第 n 到 m 筆
            ```py
            sub_df = df.iloc[n:m]
            ```
        * 欄位中包含 value
            ```py
            df.column.isin(value)
            ```
        * 判斷 Nan
            ```py
            pd.isnull(obj)  # df.isnull()
            pd.notnull(obj) # df.notnull()
            ```
        * 欄位篩選
            ```py
            new_df = df['col1'] # df.col1
            new_df = df[['col1','col2','col3']]
            df = pd.DataFrame(np.array(([1, 2, 3], [4, 5, 6])),
                        index=['mouse', 'rabbit'],
                        columns=['one', 'two', 'three'])
            new_df = df.filter(items=['one', 'three'])
            new_df = df.filter(regex='e$', axis=1)
            new_df = df.filter(like='bbi', axis=0)
            ```
    * Groupby 操作 : 
        ```py
        sub_df_obj = df.groupby(['col1'])
        ```
        * 計算各組的數量
            ```py
            sub_df_obj.size()
            ```
        * 得到各組的統計值
            ```py
            sub_df_obj.describe()
            ```
        * 根據 col1 分組後計算 col2 的統計值
            ```py
            sub_df_obj['col2'].mean()
            ```
        * 根據 col1 分組後的 col2 引用操作
            ```py
            sub_df_obj['col2'].apply(...)
            ``` 
        * 根據 col1 分組後的 col2 繪圖
            ```py
            sub_df_obj['col2'].hist()
            ```
* **Day_14 : 相關係數簡介**





