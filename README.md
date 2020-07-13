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
            * Z-transform : $ \frac{(x - mean(x))}{std(x)} $
            * Range (0 ~ 1) : $ \frac{x - min(x)}{max(x) - min(x)} $
            * Range (-1 ~ 1) : $ (\frac{x - min(x)}{max(x) - min(x)} - 0.5) * 2 $
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
* **Day_13 : 常用的 DataFrame 操作**
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
    * 想要了解兩個變數之間的**線性關係**時，相關係數是一個還不錯的簡單方法，能給出一個 -1~1 之間的值來衡量化兩個變數之間的關係。
    * Correlation Coefficient :
        $$r=\frac{1}{n-1} \sum_{i=1}^n\frac{(x_i-\bar{x})}{s_x}\frac{(y_i-\bar{y})}{s_y}$$
    * [相關係數小遊戲](http://guessthecorrelation.com/)
* **Day_15 : 相關係數實作**
    ```py
    df = pd.DataFrame(np.concatenate([np.random.randn(20).reshape(-1,1),np.random.randint(0,2,20).reshape(-1,1)],axis=1), columns=["X","y"])
    df.corr()
    np.corrcoef(df)
    ```    
    * 通常搭配繪圖進一步了解目標與變數的關係
        ```py
        df.plot(x='X',y='y',kind='scatter')
        df.boxplot(column=['X'], by=['y'])
        ```
* **Day_16 : 繪圖與樣式＆Kernel Density Estimation (KDE)**
    * 繪圖風格 : 用已經被設計過的風格，讓觀看者更清楚明瞭，包含色彩選擇、線條、樣式等。
        ```py
        plt.style.use('default') # 不需設定就會使用預設
        plt.style.use('ggplot') 
        plt.style.use('seaborn') # 採⽤ seaborn 套件繪圖
        ```
    * KDE
        * 採用無母數方法劃出觀察變數的機率密度函數
        * Density Plot 特性 :
            * 歸一 : 線下面積和為1
        * 常用的 kernal function :
            * Gaussian(Normal dist)
            * Cosine
        * 優點 : 無母數方法，對分布沒有假設
        * 缺點 : 計算量大
        * 透過 KDE Plot 可以清楚看出不同組間的分布情形
        ```py
        import matplotlib.pyplot as plt
        import seaborn as sns   

        # 將欄位分成多分進行統計繪圖
        plt.hist(df[col], edgecolor = 'k', bins = 25)

        # KDE, 比較不同的 kernel function
        sns.kdeplot(df[col], label = 'Gaussian esti.', kernel='gau')
        sns.kdeplot(adf[col], label = 'Cosine esti.', kernel='cos')
        sns.kdeplot(df[col], label = 'Triangular esti.', kernel='tri')

        # 完整分布圖 (distplot) : 將 bar 與 Kde 同時呈現
        sns.distplot(df[col])

        # 繪製 barplot 並顯示目標的 variables 
        sns.barplot(x="BIRTH_RANGE", y="TARGET", data=df)
        plt.xticks(rotation=45) # 旋轉刻度標籤
        ```
    * 延伸閱讀 :
        * [Python Graph Gallery](https://python-graph-gallery.com/)
        * [R Graph Gallery](https://www.r-graph-gallery.com/)
        * [Interactive plot，互動圖](https://bl.ocks.org/mbostock)
* **Day_17 : 把連續型變數離散化**
    * [離散化目的](https://www.zhihu.com/question/31989952) :
        * 讓事情變簡單，增加運算速度
        * 減少 outlier 對模型的影響
        * 引入非線性，提升模型表達能力
        * 提升魯棒性，減少過似合
    * 主要方法 :
        * 等寬劃分 `pd.cut()`，可使用 `np.linspace()` 進行等距切分
        * 等頻劃分 `pd.qcut()`
        * 聚類劃分
* **Day_18 : 把連續型變數離散化實作**
    * 把連續型的特徵離散化後，可以配合 `groupby` 劃出與預測目標的圖，來判斷兩者之間是否有某些關係和趨勢。
        ```py
        # 將年齡相關資料, 另外存成一個 DataFrame 來處理
        age_data = app_train[['TARGET', 'DAYS_BIRTH']]
        age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

        # 將年齡資料離散化 / 分組
        age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

        age_groups  = age_data.groupby('YEARS_BINNED').mean()

        plt.figure(figsize = (8, 8))

        # 繪製目標值平均與分組組別的長條圖
        plt.bar(range(len(age_groups.index)), age_groups['TARGET'])
        # 加上 X, y 座標說明, 以及圖表的標題
        plt.xticks(range(len(age_groups.index)), age_groups.index, rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Average Failure to Repay')
        plt.title('Failure to Repay by Age Group')
        ```
* **Day_19 : Subplot**
    * 使用時機 :
        * 有很多相似的資料要呈現時(如不同組別)
        * 同一組資料，想要同時用不同的圖呈現
        * 適時的使用有助於資訊傳達，但過度使用會讓重點混淆
    * subplot 坐標系 (列-欄-位置)
        * (321) 代表在⼀個 **3列2欄** 的最左上⾓ **列1欄1**
        * (232) 代表在一個 **2列3欄** 的 **列1欄2** 位置
        ```py
        # 方法一 : 數量少的時候或繪圖方法不同時
        plt.figure(figsize=(8,8))
        plt.subplot(321)
        plt.plot([0,1],[0,1], label = 'I am subplot1')
        plt.legend()
        plt.subplot(322)
        plt.plot([0,1],[1,0], label = 'I am subplot2')
        plt.legend()

        # 方法二 : 數量多的時候或繪圖方法雷同時
        nrows = 5
        ncols = 2
        plt.figure(figsize=(10,30))
        for i in range(nrows*ncols):
            plt.subplot(nrows, ncols, i+1)
        ```
    * 延伸閱讀 :
        * [matplotlib 官⽅方範例例](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)
        * [複雜版 subplot 寫法](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
        * [另類⼦子圖 Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)
* **Day_20 : Heatmap & Grid-plot**
    * heatmap
        * 常用於呈現變數間的相關性
        * 也可以用於呈現不同條件下的數量關係
        * 常用於呈現混淆矩陣(Confusion matrix)
        ```py
        plt.figure(figsize = (8, 6))
        # 繪製相關係數 (correlations) 的 Heatmap
        sns.heatmap(df.corr(), cmap = plt.cm.RdYlBu_r, vmin = -1.0, annot = True, vmax = 1.0)
        plt.title('Correlation Heatmap')
        ```
    * pairplot
        * 對角線 : 該變數的分布(distribution)
        * 非對角線 : 倆倆變數間的散佈圖
        ```py
        import seaborn as sns; sns.set(style="ticks", color_codes=True)
        iris = sns.load_dataset("iris")
        g = sns.pairplot(iris)
        ```
    * PairGrid
        * 可以自訂對角線和非對角線的繪圖類型
        ```py
        g = sns.PairGrid(iris, hue="species")
        g = g.map_diag(plt.hist)    # 對角線繪圖類型
        g = g.map_offdiag(plt.scatter)  # 非對角線繪圖類型
        g = g.add_legend()

        g = sns.PairGrid(iris)
        g = g.map_upper(sns.scatterplot)    # 上三角繪圖類型
        g = g.map_lower(sns.kdeplot, colors="C0")   # 下三角繪圖類型
        g = g.map_diag(sns.kdeplot, lw=2)
        ```
    * 延伸閱讀 :
        * [基本 Heatmap](https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html)
        * [進階 Heatmap](https://www.jianshu.com/p/363bbf6ec335)
        * [pairplot 更多應用](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)
* **Day_21 : 模型初體驗 - Logistic Regression**
    * [Logistic Regression](https://www.youtube.com/playlist?list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy)
        ```py
        from sklearn.linear_model import LogisticRegression

        # 設定模型與模型參數
        log_reg = LogisticRegression(C = 0.0001)

        # 使用 Train 資料訓練模型
        log_reg.fit(train, train_labels)

        # 用模型預測結果
        # 請注意羅吉斯迴歸是分類預測 (會輸出 0 的機率, 與 1 的機率), 而我們只需要留下 1 的機率這排
        log_reg_pred = log_reg.predict_proba(test)[:, 1]

        # 將 DataFrame/Series 轉成 CSV 檔案方法
        df['Target'] = log_reg_pred
        df.to_csv(file_name, encoding='utf-8', index=False)
        ```
### 資料科學與特徵工程技術 Data Science & Feature Engineering
* **Day_22 : 特徵工程簡介**
    * 資料工程是將**事實**對應到**分數**的**轉換**
    * 由於資料包含類別特徵(文字)和數值特徵，所以最小的特徵工程至少包含一種**類別編碼**(例如:標籤編碼)和**特徵縮放**方法(例如:最小最大化)
        ```py
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler

        LEncoder = LabelEncoder()
        MMEncoder = MinMaxScaler()
        for c in df.columns:
            df[c] = df[c].fillna(-1)
            if df[c].dtype == 'object':
                df[c] = LEncoder.fit_transform(list(df[c].values))
            df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
        ```
    * 延伸閱讀 : [特徵工程是什麼](https://www.zhihu.com/question/29316149)
* **Day_23 : 數值型特徵 - 去除偏態**
    * 當**離群值**資料比例太高，或者**平均值沒有代表性**時，可以考慮去除偏態
    * 去除偏態包含 : 對數去偏(log1p)、方根去偏(sqrt)、分布去偏(boxcox)
    * 使用 box-cox 分不去偏時，除了注意 $\lambda$ 參數要藉於 0 到 0.5 之間，並且要注意轉換前的數值不可小於等於 0
        ```py
        # 對數去偏
        df_fixed['Fare'] = np.log1p(df_fixed['Fare'])

        # 方根去偏
        df['score'] = np.sqrt(df['score']) * 10

        from scipy import stats
        # 修正方式 : 加入下面這一行, 使最小值大於 0, 類似log1p的概念
        df_fixed['Fare'] = df_fixed['Fare'] + 1
        df_fixed['Fare'] = stats.boxcox(df_fixed['Fare'])[0]
        ```
    * 延伸閱讀 : [偏度與峰度](https://blog.csdn.net/u013555719/article/details/78530879)
* **Day_24 : 類別型特徵 - 基礎處理**
    * 類別型特徵有**標籤編碼**(Label Encoding)與**獨熱編碼**(One Hot Encoding)兩種基礎編碼方式
    * 標籤編碼將特徵依序轉為代碼，若特徵沒有大小順序之別，則大小順序沒有意義，常用於非深度學習模型，深度學習模型主要依賴倒傳導，標籤編碼不易收斂
    * 當特徵重要性高且可能值少時，可考慮獨熱編碼
        ```py
        from sklearn.preprocessing import LabelEncoder
        
        object_features = []
        for dtype, feature in zip(df.dtypes, df.columns):
            if dtype == 'object':
                object_features.append(feature)

        df = df[object_features]
        df = df.fillna('None')
        # 標籤編碼
        for c in df.columns:
            df_temp[c] = LabelEncoder().fit_transform(df[c])
        # 獨熱編碼
        df_temp = pd.get_dummies(df)
        ```
    * 延伸閱讀 : [標籤編碼與獨熱編碼](https://blog.csdn.net/u013555719/article/details/78530879)
* **Day_25 : 類別型特徵 - 均值編碼**
    * 均值編碼(Mean Encoding) : 使用目標值的平均值取代原本類別型特徵
    * 當類別特徵與目標明顯相關時，該考慮採用均值編碼
    * 樣本數少時可能是極端值，平均結果可能誤差很大，需使用平滑公式來調整
        * 當平均值可靠度低則傾向相信總平均
        * 當平均值可靠性高則傾向相信類別的平均
        * 依照紀錄的比數，在兩者間取折衷
    $$新類別均值 = \frac{原類別平均*類別樣本數 + 全部的總平均*調整因子}{類別樣本數 + 調整因子}$$
    * 相當容易 overfitting 請小心使用
        ```py
        data = pd.concat([df[:train_num], train_Y], axis=1)
        for c in df.columns:
            mean_df = data.groupby([c])['target'].mean().reset_index()
            mean_df.columns = [c, f'{c}_mean']
            data = pd.merge(data, mean_df, on=c, how='left')
            data = data.drop([c] , axis=1)
        data = data.drop(['target'] , axis=1)
        ```
* **Day_26 : 類別型特徵 - 其他進階處理**
    * 計數編碼(Counting) : 計算類別在資料中出現次數，當目前平均值與類別筆數呈現正/負相關時，可以考慮使用
        ```py
        count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
        df = pd.merge(df, count_df, on=['Ticket'], how='left')
        ```
    * 雜湊編碼 : 將類別由雜湊函數對應到一組數字
        * 調整雜湊函數對應值的數量，在計算空間/時間與鑑別度間取折衷
        * 提高訊息密度並減少無用的標籤
        ```py
        df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)
        ```
    * 雜湊編碼也不佳時可使用嵌入式編碼(Embedding)，但需要基於深度學習前提下
    * 延伸閱讀 :
        * [特徵哈希](https://blog.csdn.net/laolu1573/article/details/79410187)
        * [文本特徵抽取](https://www.jianshu.com/p/063840752151)
* **Day_27 : 時間型特徵**
    * 時間型特徵分解 :
        * 依照原意義分欄處理，年、月、日、時、分、秒或加上第幾周和星期幾
        * 週期循環特徵
            * 年週期 : 與季節溫度相關
            * 月週期 : 與薪水、繳費相關
            * 周週期 : 與周休、消費習慣相關
            * 日週期 : 與生理時鐘相關
        * 週期數值除了由欄位組成還需**頭尾相接**，因此一般以**正(餘)弦函數**加以組合
            * 年週期 : ( 正 : 冷 / 負 : 熱 )
                $cos((月/6 + 日/180)\pi)$
            * 周週期 : ( 正 : 精神飽滿 / 負 : 疲倦 )
                $sin((星期幾/3.5 + 小時/84)\pi)$
            * 日週期 : ( 正 : 精神飽滿 / 負 : 疲倦 )
                $sin((小時/12 + 分/720 + 秒/43200)\pi)$
            * 須注意最高點與最低點的設置
        ```py
        import datetime

        # 時間特徵分解方式:使用datetime
        df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
        df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
        df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
        df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
        df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
        df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
        df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')

        # 加入星期幾(day of week)和第幾周(week of year)
        df['pickup_dow'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%w')).astype('int64')
        df['pickup_woy'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%W')).astype('int64')

        # 加上"日週期"特徵 (參考講義"週期循環特徵")
        import math
        df['day_cycle'] = df['pickup_hour']/12 + df['pickup_minute']/720 + df['pickup_second']/43200
        df['day_cycle'] = df['day_cycle'].map(lambda x:math.sin(x*math.pi))

        # 加上"年週期"與"周週期"特徵
        df['year_cycle'] = df['pickup_month']/6 + df['pickup_day']/180
        df['year_cycle'] = df['year_cycle'].map(lambda x:math.cos(x*math.pi))
        df['week_cycle'] = df['pickup_dow']/3.5 + df['pickup_hour']/84
        df['week_cycle'] = df['week_cycle'].map(lambda x:math.sin(x*math.pi))
        ```
    * 延伸閱讀 :
        * [時間日期處理](http://www.wklken.me/posts/2015/03/03/python-base-datetime.html)
        * [datetime](https://docs.python.org/3/library/datetime.html)
* **Day_28 : 特徵組合 - 數值與數值組合**
    * 除了基本的加減乘除，最關鍵的是**領域知識**，例如將經緯度資料組合成高斯距離
        ```py
        # 增加緯度差, 經度差, 座標距離等三個特徵
        df['longitude_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
        df['latitude_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
        df['distance_2D'] = (df['longitude_diff']**2 + df['latitude_diff']**2)**0.5

        import math
        latitude_average = df['pickup_latitude'].mean()
        latitude_factor = math.cos(latitude_average/180*math.pi)
        df['distance_real'] = ((df['longitude_diff']*latitude_factor)**2 + df['latitude_diff']**2)**0.5
        ```
    * 機器學習的關鍵是特徵工程，能有效地提升模型預測能力            
    * 延伸閱讀 : [特徵交叉](https://segmentfault.com/a/1190000014799038)
* **Day_29 : 特徵組合 - 類別與數值組合**
    * 群聚編碼(Group by Encoding) : 類別特徵與數值特徵可以使用群聚編碼組合出新的特徵
        * 常見的組合方式有 `mean`,`mdian`,`mode`,`max`,`min`,`count`
        * 與均值編碼(Mean Encoding)的比較
            | 名稱                  | 均值編碼 Encoding | 群聚編碼 Group by Encoding |
            |-----------------------|------------------|---------------------------|
            | 平均對象                | 目標值           | 其他數值型特徵                |
            | 過擬合\(Overfitting\)  | 容易            | 不容易                    |
            | 對均值平滑化\(Smoothing\) | 需要            | 不需要                    |
        * 機器學習的特徵是 **寧爛勿缺** 的，以前非樹狀模型為了避免共線性，會希望類似特徵不要太多，但現在強力模型大多是樹狀模型，所以通通做成~~雞精~~特徵
    * 延伸閱讀 : [數據聚合與分組](https://zhuanlan.zhihu.com/p/27590154)
        ```py
        # 取船票票號(Ticket), 對乘客年齡(Age)做群聚編碼
        df['Ticket'] = df['Ticket'].fillna('None')
        df['Age'] = df['Age'].fillna(df['Age'].mean())

        mean_df = df.groupby(['Ticket'])['Age'].mean().reset_index()
        mode_df = df.groupby(['Ticket'])['Age'].apply(lambda x: x.mode()[0]).reset_index()
        median_df = df.groupby(['Ticket'])['Age'].median().reset_index()
        max_df = df.groupby(['Ticket'])['Age'].max().reset_index()
        min_df = df.groupby(['Ticket'])['Age'].min().reset_index()
        temp = pd.merge(mean_df, mode_df, how='left', on=['Ticket'])
        temp = pd.merge(temp, median_df, how='left', on=['Ticket'])
        temp = pd.merge(temp, max_df, how='left', on=['Ticket'])
        temp = pd.merge(temp, min_df, how='left', on=['Ticket'])
        temp.columns = ['Ticket', 'Age_Mean', 'Age_Mode', 'Age_Median', 'Age_Max', 'Age_Min']
        temp.head()
        ```
        | Index | Ticket | Age\_Mean  | Age\_Mode  | Age\_Median | Age\_Max | Age\_Min   |
        |-------|--------|------------|------------|-------------|----------|------------|
        | 0     | 110152 | 26\.333333 | 16\.000000 | 30\.000000  | 33\.0    | 16\.000000 |
        | 1     | 110413 | 36\.333333 | 18\.000000 | 39\.000000  | 52\.0    | 18\.000000 |
        | 2     | 110465 | 38\.349559 | 29\.699118 | 38\.349559  | 47\.0    | 29\.699118 |
        | 3     | 110564 | 28\.000000 | 28\.000000 | 28\.000000  | 28\.0    | 28\.000000 |
        | 4     | 110813 | 60\.000000 | 60\.000000 | 60\.000000  | 60\.0    | 60\.000000 |
* **Day_30 : 特徵選擇**
    * 特徵需要適當的增加與減少，以提升精確度並減少計算時間
        * 增加特徵 : 特徵組合(Day_28)，群聚編碼(Day_29)
        * 減少特徵 : 特徵選擇(Day_30)
    * 特徵選擇有三大類方法
        * 過濾法(Filter) : 選定統計值與設定門檻，刪除低於門檻的特徵
        * 包裝法(Wrapper) : 根據目標函數，逐步加入特徵或刪除特徵
        * 嵌入法(Embedded) : 使用機器學習模型，根據擬合後的係數，刪除係數低餘門檻的特徵
    * 相關係數過濾法
        ```py
        # 計算df整體相關係數, 並繪製成熱圖
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr = df.corr()
        sns.heatmap(corr)
        plt.show()

        # 篩選相關係數大於 0.1 或小於 -0.1 的特徵
        high_list = list(corr[(corr['SalePrice']>0.1) | (corr['SalePrice']<-0.1)].index)
        # 刪除目標欄位
        high_list.pop(-1)
        ```
    * Lasso(L1)嵌入法
        * 使用 Lasso Regression 時，調整不同的正規化程度，就會自然使得一部分特徵係數為 0，因此刪除係數是 0 的特徵，不須額外指定門檻，但需調整正規化程度
        ```py
        from sklearn.linear_model import Lasso
        L1_Reg = Lasso(alpha=0.001)
        train_X = MMEncoder.fit_transform(df)
        L1_Reg.fit(train_X, train_Y)
        L1_Reg.coef_

        L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
        df.columns[L1_mask] # index type

        from itertools import compress
        L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
        L1_list = list(compress(list(df), list(L1_mask)))   # list type
        ```
    * GDBT(梯度提升樹)嵌入法
        * 使用梯度提升樹擬合後，以特徵在節點出現的頻率當作特徵重要性，以此刪除重要性低於門檻的特徵

            |           | 計算時間 | 共線性  | 特徵穩定性 |
            |-----------|------|------|-------|
            | 相關係數過濾法   | 快速   | 無法排除 | 穩定    |
            | Lasso 嵌入法 | 快速   | 能排除  | 不穩定   |
            | GDBT 嵌入法  | 較慢   | 能排除  | 穩定    |
    * 延伸閱讀 :
        * [特徵選擇](https://zhuanlan.zhihu.com/p/32749489)
        * [特徵選擇手冊](https://machine-learning-python.kspax.io/intro-1)
* **Day_31 : 特徵評估**
    * 特徵的重要性 : 分支次數、特徵覆蓋度、損失函數降低量
    * sklearn 當中的樹狀模型，都有特徵重要性這項方法 `.feature_importance_`，而實際上都是分支次數
        ```py
        from sklearn.ensemble import RandomForestRegressor
        # 隨機森林擬合後, 將結果依照重要性由高到低排序
        estimator = RandomForestRegressor()
        estimator.fit(df.values, train_Y)
        # estimator.feature_importances_ 就是模型的特徵重要性, 這邊先與欄位名稱結合起來, 才能看到重要性與欄位名稱的對照表
        feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
        feats = feats.sort_values(ascending=False)
        ```
    * 進階版的 GDBT 模型(`Xgboost`、`lightbm`、`catboost`)中，才有上述三種不同的重要性
        |                | Xgboost 對應參數 | 計算時間 | 估計精確性 | sklearn 有此功能 |
        |----------------|--------------|------|-------|--------------|
        | 分支次數           | weight       | 最快   | 最低    | O            |
        | 分支覆蓋度          | cover        | 快    | 中     | X            |
        | 損失降低量\(資訊增益度\) | gain         | 較慢   | 最高    | X            |
    * 機器學習的優化循環
        1. 原始特徵
        2. 進階版 GDBT 模型擬合
        3. 用特徵重要性增刪特徵
            * 特徵選擇(刪除) : 挑選門檻，刪除一部分重要性較低的特徵
            * 特徵組合(增加) : 依領域知識，對前幾名的特徵做特徵組合或群聚編碼，形成更強力特徵
        4. 交叉驗證(cross validation)，確認特徵效果是否改善
    * 排序重要性(Permutation Importance)
        * 雖然特徵重要性相當食用，然而計算原理必須基於樹狀模型，於是有了可延伸至非樹狀模型的排序重要性
        * 排序重要性是打散單一特徵的資料排序，再用原本模型重新預測，觀察打散前後誤差變化有多少

            |        | 特徵重要性 Feature Importance | 排序重要性 Permutation Importance |
            |--------|--------------------------|------------------------------|
            | 適用模型   | 限定樹狀模型                   | 機器學習模型均可                     |
            | 計算原理   | 樹狀模型的分歧特徵                | 打散原始資料中單一特徵的排序               |
            | 額外計算時間 | 較短                       | 較長                           |
    * 延伸閱讀 :
        * [特徵選擇的優化流程](https://juejin.im/post/5a1f7903f265da431c70144c)
        * [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights)
* **Day_32 : 分類型特徵優化 - 葉編碼**
    * 葉編碼(leaf encoding) : 採用決策樹的葉點作為編碼依據重新編碼
        * 葉編碼的目的是**重新標計**資料，以擬合後的樹狀模型分歧條件，將資料**離散化**，這樣比人為寫作的判斷條件更精準，更符合資料的分布情形
        * 葉編碼完後，因特徵數量較多，通常搭配**羅吉斯回歸**或者**分解機**做預測，其他模型較不適合
        ```py
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        # 因為擬合(fit)與編碼(transform)需要分開, 因此不使用.get_dummy, 而採用 sklearn 的 OneHotEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import roc_curve
        # 隨機森林擬合後, 再將葉編碼 (*.apply) 結果做獨熱 / 邏輯斯迴歸
        rf = RandomForestClassifier(n_estimators=20, min_samples_split=10, min_samples_leaf=5, 
                            max_features=4, max_depth=3, bootstrap=True)
        onehot = OneHotEncoder()
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)

        rf.fit(train_X, train_Y)
        onehot.fit(rf.apply(train_X))
        lr.fit(onehot.transform(rf.apply(val_X)), val_Y)

        # 將隨機森林+葉編碼+邏輯斯迴歸結果輸出
        pred_rf_lr = lr.predict_proba(onehot.transform(rf.apply(test_X)))[:, 1]
        fpr_rf_lr, tpr_rf_lr, _ = roc_curve(test_Y, pred_rf_lr)
        # 將隨機森林結果輸出
        pred_rf = rf.predict_proba(test_X)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(test_Y, pred_rf)
        ```
    * 延伸閱讀 :
        * [Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)
        * [Algorithm-GBDT Encoder](https://zhuanlan.zhihu.com/p/31734283)
        * [分解機，Factorization Machine，FM](https://kknews.cc/code/62k4rml.html)
* **Day_33 : 機器如何學習**
    * 定義模型 : 線性回歸、決策樹、神經網路等等
        * 例如線性回歸 : $ y = b + w * x $
            * $w$ : weight 和 $b$ : bias 就是模型參數
            * 不同參數模型會產生不同的 $\hat{y}$
            * 希望產生出來的 $\hat{y}$ 與真實答案 $y$ 越接近越好
            * 找出一組參數讓模型產生的 $\hat{y}$ 與真正的 $y$ 很接近，這個過程有點像是學習的概念。
    * 評估模型好壞 : 定義一個**目標函數**(objective function)也可稱為**損失函數**(Loss function)，來衡量模型好壞
        * 例如線性回歸可以使用**均方差**(mean square error)來衡量
            $$ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2$$
        * Loss 越大代表模型預測愈不准，代表不該選擇這個參數
    * 找出最佳參數 : 可以使用爆力法、梯度下降(Gradient Descent)、增量訓練(Addtive Training)等方式
        * 過擬合(over-fitting) : 訓練過程學習到了噪音導致在實際應用失準
        * 欠擬合(under-fitting) : 模型無法好好的擬合訓練數據
            * 如何知道 : 觀察訓練資料與測試資料的誤差趨勢
            * 如何改善 :
                * 過擬合 : 
                    * 增加資料量
                    * 降低模型複雜度
                    * 使用正規化(Regularization)
                * 欠擬合 :
                    * 增加模型複雜度
                    * 減輕或不使用正規化
    * 延伸閱讀 : [學習曲線與 bias/variance trade-off](http://bangqu.com/yjB839.html)
* **Day_34 : 訓練與測試集切分**
    * 為何需要切分 :
        * 機器學習模型需要資料訓練
        * 若所有資料都送進訓練模型，就沒有額外資料來評估模型
        * 機器模型可能過擬合，需要驗證/測試集來評估模型是否過擬合
    * 使用 [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 進行切分
        ```py
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
        ```
    * [K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) Cross Validation : 若僅做一次切分，有些資料會沒有被拿來訓練，因此有 cross validation 方法，讓結果更穩定
        ```py
        from sklearn.model_selection import KFold
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([1, 2, 3, 4])
        kf = KFold(n_splits=2, shuffle=False)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        ```
    * 驗證集與測試集差異 : 驗證集常用來評估不同超參數或不同模型的結果，測試集則是預先保留的資料，在專案開發過程中都不使用，最終在拿來做測試
    * 延伸閱讀 : [訓練、驗證與測試集的意義](https://www.youtube.com/watch?v=D_S6y0Jm6dQ&feature=youtu.be&t=1948)
* **Day_35 : Regression & Classification**
    * 機器學習主要分為**回歸**問題與**分類**問題
        * 回歸問題 : 預測目標值為實數($-\infty$至$\infty$)(continuous)
        * 分類問題 : 預測目標值為類別(0或1)(discrete)
        * 回歸問題可以轉化為分類問題 :
            * 原本預測身高(cm)問題可以轉化為預測高、中等、矮(類別)
    * 二元分類(binary-class) vs. 多元分類(multi-class)
        * 二元分類 : 目標類別只有兩個，例如詐騙分析(正常用戶、異常用戶)、瑕疵偵測(瑕疵、正常)
        * 多元分類 : 目標類別有兩種以上，例如手寫數字辨識(0~9)，影像競賽(ImageNet)有高達1000個類別
    * 多元分類 vs. 多標籤(multi-label)
        * 多元分類 : 每個樣本只能歸在一個類別
        * 多標籤 : 一個樣本可以屬於多個類別
    * 延伸閱讀 : 
        * [回歸與分類比較](http://zylix666.blogspot.com/2016/06/supervised-classificationregression.html)
        * [Multi-class vs. Multi-label ](https://medium.com/coinmonks/multi-label-classification-blog-tags-prediction-using-nlp-b0b5ee6686fc)
* **Day_36 : 評估指標選定**
    * 設定各項指標來評估模型的準確性，最常見的為準確率(**Accuracy**) = 正確認類樣本數/總樣本數
    * 不同的評估指標有不同的評估準則與面向，衡量的重點有所不同
    * 評估指標 - 回歸 : 觀察預測值(prediction)與實際值(ground truth)的**差距**
        * MAE(mean absolute error)，範圍[0,inf]
        * MSE(mean square error)，範圍[0,inf]
        * R-square，範圍[0,1]
    * 評估指標 - 分類 : 觀察預測值與實際值的**正確程度**
        * AUC(area under curve)，範圍[0,1]
        * F1-score(precision, recall)，範圍[0,1]
        * 混淆矩陣(Confusion Matrix)
    * 回歸問題可透過 R-square 快速了解準確度，二元分類問題通常使用 AUC 評估，希望哪個類別不要分錯則可使用 F1-score 並觀察 precision 與 recall 數值，多分類問題則可使用 top-k accuracy，例如 ImageNet 競賽通常採用 top-5 accuracy
    * Q&A :
        * AUC 計算怪怪的，AUC 的 y_pred 的值填入每個樣本預測機率(probility)而非分類結果
        * F1-score 計算則填入每個樣本分類結果，如機率 >= 0.5 則視為 1，而非填入機率值
        ```py
        from sklearn import metrics, datasets
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        # MAE, MSE, R-square
        X, y = datasets.make_regression(n_features=1, random_state=42, noise=4) # 生成資料
        model = LinearRegression() # 建立回歸模型
        model.fit(X, y) # 將資料放進模型訓練
        prediction = model.predict(X) # 進行預測
        mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
        mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
        r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估

        # AUC
        cancer = datasets.load_breast_cancer() # 我們使用 sklearn 內含的乳癌資料集
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=50, random_state=0)
        y_pred = np.random.random((50,)) # 我們先隨機生成 50 筆預測值，範圍都在 0~1 之間，代表機率值
        auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。 **這邊特別注意 y_pred 必須要放機率值進去!**

        # F1-score, precision, recall
        threshold = 0.5 
        y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0
        f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
        precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
        recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
        ```

    * 延伸閱讀 :
        * [超詳解 AUC](https://www.dataschool.io/roc-curves-and-auc-explained/)
        * [更多評估指標](https://zhuanlan.zhihu.com/p/30721429)
* **Day_37 : Regression 模型**
    * Linear Regression (線性回歸) : 簡單線性模型，可用於回歸問題，須注意[資料共線性與資料標準化問題](https://blog.csdn.net/Noob_daniel/article/details/76087829)，通常可做為 baseline 使用
    * Logistic Regression (羅吉斯回歸) : 分類模型，將線性模型結果加上 [sigmoid](https://baike.baidu.com/item/Sigmoid%E5%87%BD%E6%95%B0/7981407) 函數，將預測值限制在 0~1 之間，即為預測機率值
    * 延伸閱讀 :
        * [Andrew Ng 教你 Linear Regression](https://zh-tw.coursera.org/lecture/machine-learning/model-representation-db3jS)
        * [Logistic Regression 數學原理](https://blog.csdn.net/qq_23269761/article/details/81778585)
        * [Linear Regression 詳細介紹](https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_linear_regression_works.html)
        * [Logistic Regression 詳細介紹](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
        * [你可能不知道的 Logistic Regression](https://taweihuang.hpd.io/2017/12/22/logreg101/)
* **Day_38 : Regression 模型 </>**
    * 使用 scikit-learn 套件
        ```py
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import mean_squared_error, accuracy_score
        # 線性回歸
        reg = LinearRegression().fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        # 羅吉斯回歸
        logreg = LogisticRegression().fit(x_train, y_train)
        y_pred = logreg.predict(x_test)
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        ```
    * Logistic Regression 參數
        * Penalty : 使用 "L1" or "L2" 正則化參數
        * C : 正則化的強度，數字越小模型越簡單
        * Solver : 對損失函數的優化方法，詳細參考[連結](https://blog.csdn.net/lc574260570/article/details/82116197)
        * Multi-class : 選擇 one-vs-rest 或 multi-nominal 分類方式，若有 10 class， ovr 是訓練 10 個二分類模型，第一個模型負責分類 (class1, non-class1)；第二個負責(class2, non-class2)，以此類推。multi-nominal 是直接訓練多分類模型。詳細參考[連結](https://www.quora.com/What-is-the-difference-between-one-vs-all-binary-logistic-regression-and-multinomial-logistic-regression)
    * 延伸閱讀 :
        * [更多 Linear regression 和 Logistic regression 範例](https://github.com/trekhleb/homemade-machine-learning)
        * [深入了解 multi-nominal Logistic Regresson 原理](http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)
* **Day_39 : LASSO, Ridge regression**
    * 機器學習模型的目標函數有兩個非常重要的元素
        * 損失函數 (Loss function) : 衡量實際值與預測值差異，讓模型往正確方向學習
        * 正則化 (Regularization) : 避免模型過於複雜，造成過擬合
        * 為了避免過擬合我們把正則化加入目標函數，目標函數 = 損失函數 + 正則化
        * 正則化可以懲罰模型的複雜度，當模型越大其值越大
    * 正則化函數 : 用來衡量模型的複雜度
        * L1 : $\alpha\sum|weight|$
        * L2 : $\alpha\sum(weight)^2$
        * 這兩種都是希望模型的參數數值不要太大，原因是參參數的數值變小，噪音對最終輸出的結果影響越小，提升模型的泛化能力，但也讓模型的擬合能力下降
    * LASSO 為 Linear Regression 加上 L1
    * Ridge 為 Linear Regression 加上 L2
    * 其中有個超參數 $\alpha$ 可以調整正則化強度 
    * 延伸閱讀 : 
        * [Linear, Lasso, Ridge Regression 本質區別](https://www.zhihu.com/question/38121173)
        * [PCA 與 Ridge regression 的關係](https://taweihuang.hpd.io/2018/06/10/pca-%E8%88%87-ridge-regression-%E7%9A%84%E9%97%9C%E4%BF%82/
        )
* **Day_40 : LASSO, Ridge regression </>**
    ```py
    from sklearn import datasets
    from sklearn.linear_model import Lasso, Ridge
    # 讀取糖尿病資料集
    diabetes = datasets.load_diabetes()
    # 切分訓練集/測試集
    x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=4)
    
    # 建立一個線性回歸模型
    lasso = Lasso(alpha=1.0)
    # 將訓練資料丟進去模型訓練
    lasso.fit(x_train, y_train)
    # 將測試資料丟進模型得到預測結果
    y_pred = lasso.predict(x_test)
    # 印出訓練後的模型參參數
    print(lasso.coef_)

    # 建立一個線性回歸模型
    ridge = Ridge(alpha=1.0)
    # 將訓練資料丟進去模型訓練
    ridge.fit(x_train, y_train)
    # 將測試資料丟進模型得到預測結果
    y_pred = regr.predict(x_test)
    # 印出訓練後的模型參參數
    print(ridge.coef_)
    ```
* **Day_41 : 決策樹 Decision Tree**
    * 決策樹 (Decision Tree) : 透過一系列的是非問題，幫助我們將資料切分，可是視覺化每個切分過程，是個具有非常高解釋性的模型
        * 從訓練資料中找出規則，讓每一次決策使訊息增益 (information Gain) 最大化
        * 訊息增益越大代表切分後的兩群，群內相似度越高，例如使用健檢資料來預測性別，若使用頭髮長度 50 公分進行切分，則切分後的兩群資料很有可能多數為男生或女生(相似程度高)，這樣頭髮長度就是個好 feature
    * 如何衡量相似程度
        * 吉尼係數(不純度) (gini-index)
            $$Gini = 1 - \sum_jp_j^2$$
        * 熵 (entropy)
            $$Entropy = -\sum_jp_jlog_2p_j$$
    * 決策樹的特徵重要性
        * 我們可以從構建樹的過程中，透過 feature 被⽤用來切分的次數，來得知哪些 features 是相對有用的
        * 所有 feature importance 的總和為 1
        * 實務上可以使用 feature importance 來了解模型如何進行分類
    * 延伸閱讀 :
        * [決策樹運作](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)
        * [決策樹與回歸問題](https://www.saedsayad.com/decision_tree_reg.htm)
* **Day_42 : 決策樹 </>**
    * 機器學習的建模步驟 :
        1. 讀取和檢查資料
            * 使用 pandas 讀取 .csv 檔 : pd.read_csv
            * 使用 numpy 讀取 txt 檔 : np.loadtxt
            * 使用 sklearn 內建資料集 : sklearn.datasets.load_xxx
            * 檢查資料數量 : data.shape
        2. 將資料切分為訓練 (train) 與測試集 (test)
            * train_test_split(data)
        3. 建立模型開始訓練 (fit)
            * clf = DecisionTreeClassifier()
            * clf.fit(x_train, y_train)
        4. 將測試資料放進訓練好的模型進行預測 (predict)，並測試資料的 label (y_test) 做評估
            * clf.predict(x_test)
            * accuracy_score(y_test, y_pred)
            * f1_score(y_test, y_pred)
    * 根據回歸/分類問題建立不同的 Classifier
        ```py
        from sklearn.tree_model import DecisionTreeRegressor
        from sklearn.tree_model import DecisionTreeClassifier
        
        # 讀取鳶尾花資料集
        iris = datasets.load_iris()
        # 切分訓練集/測試集
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)
        # 建立模型
        clf = DecisionTreeClassifier()
        # 訓練模型
        clf.fit(x_train, y_train)
        # 預測測試集
        y_pred = clf.predict(x_test)
        print("Acuuracy: ", metrics.accuracy_score(y_test, y_pred))
        # 列出特徵和重要性
        print(iris.feature_names)
        print("Feature importance: ", clf.feature_importances_)
        ```
    * 決策樹的超參數
        * Criterion: 衡量資料相似程度的 metric
        * Max_depth: 樹能生長的最深限制
        * Min_samples_split: 至少要多少樣本以上才進行切分
        * Min_samples_lear: 最終的葉子(節點)上至少要有多少樣本
        ```py
        clf = DecisionTreeClassifier(
                criterion = 'gini',
                max_depth = None,
                min_samples_split = 2,
                min_samples_left = 1,
        )
        ```
    * 延伸閱讀 : [Creating and Visualizing Decision Trees with Python](https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176)
* **Day_43 : 隨機森林樹 Random Forest**
    * 決策樹缺點 :
        * 若不對決策樹進行限制(樹深度、葉子上至少要有多少樣本等)，決策樹非常容易 over-fitting
    * 集成模型 - 隨機森林 (Random Forest)
        * 集成 (Ensemble) : 將多個模型的結果組合在一起，透過**投票**或是**加權**的方式獲得最終結果
        * 每棵樹使用部分訓練資料與特徵進行訓練而成
    * 延伸閱讀 :
        * [隨機森林](http://hhtucode.blogspot.com/2013/06/ml-random-forest.html)
        * [How Random Forest Algorithm Works](https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674)
        * [bootstrap](http://sofasofa.io/forum_main_post.php?postid=1000691)
* **Day_44 : 隨機森林樹 </>**
    ```py
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestClassifier(
            n_estimators=10, #決策樹的數量量
            criterion="gini",
            max_features="auto", #如何選取 features         
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1)
    # 將訓練資料丟進去模型訓練
    clf.fit(x_train, y_train)
    # 將測試資料丟進模型得到預測結果
    y_pred = clf.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # 建立一個線性回歸模型
    regr = RandomForestRegressor()
    # 將訓練資料丟進去模型訓練
    regr.fit(x_train, y_train)
    # 將測試資料丟進模型得到預測結果
    y_pred = regr.predict(x_test)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    ```
* **Day_45 : 梯度提升機 Gradient Boosting Machine**
    * 隨機森林使用的方法為 Bagging (Bootstrap Aggregating)，用抽樣資料與特徵生成每一棵樹，最後在取平均
    * Boosting 是另一種集成方法，希望由後面生成的樹來修正前面學不好的地方
    * Bagging :
        * 透過抽樣 (sampling) 方式生成每一棵樹，樹與樹之間是獨立的
        * 降低 over-fitting
        * 減少 variance
        * Independent classifiers
    * Boosting :
        * 透過序列 (additive) 方式生成每一棵樹，每棵樹與前面的樹關聯
        * 可能會 over-fitting
        * 減少 bias 和 variance
        * Sequential classifiers
    * 延伸閱讀 :
        * [梯度提升決策樹](https://ifun01.com/84A3FW7.html)
        * [XGboost](https://www.youtube.com/watch?v=ufHo8vbk6g4)
        * [陳天奇 - Boosted Tree](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
        * [李弘毅 - Ensemble](https://www.youtube.com/watch?v=tH9FH1DH5n0)
* **Day_46 : 梯度提升機 </>**
    ```py
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor

    clf = GradientBoostingClassifier(
        loss="deviance", #Loss 的選擇，若若改為 exponential 則會變成Adaboosting 演算法，概念念相同但實作稍微不同
        learning_rate=0.1, #每棵樹對最終結果的影響，應與 n_estimators 成反比
        n_estimators=100 #決策樹的數量量
        )
    # 訓練模型
    clf.fit(x_train, y_train)
    # 預測測試集
    y_pred = clf.predict(x_test)
    print("Acuuracy: ", metrics.accuracy_score(y_test, y_pred))
    ```
    * 延伸閱讀 : [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
* **Day_47 : 超參數調整**
    * 機器學習中的超參數
        * LASSO, Ridge : $\alpha$ 的大小
        * 決策樹 : 樹的深度、節點最小樣本數
        * 隨機森林 : 樹的數量
    * 超參數會影響模型訓練的結果，建議先使用預設再慢慢進行調整
    * 超參數會影響結果但提升效果有限，資料清理和特徵工程才能最有效的提升準確率
    * 超參數的調整方法
        * 窮舉法 (Grid Search) : 直接指定超參數的範圍組合，每一組參數都訓練完成，再根據驗證集的結果選擇最佳參數
        * 隨機搜尋 (Random Search) : 指定超參數範圍，用均勻分布進行參數抽樣，用抽到的參數進行訓練，再根據驗證集的結果選擇最佳參數，[隨機搜尋](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881)通常能獲得較好的結果
    * 正確的超參數調整步驟 : 若使用同意分驗證集 (validation) 來調參，可能讓模型過於擬合驗證集，正確步驟是使用 Cross-validation 確保模型的泛化性
        1. 將資料切分為訓練/測試集，測試集先保留不用
        2. 將剛切好的訓練集，再使用 Cross-validation 切成 K 份訓練/驗證集
        3. 用 gird/random search 的超參數進行訓練與評估
        4. 選出最佳參數，用該參數與全部訓練集建模
        5. 最後使用測試集評估結果
        ```py
        from sklearn import datasets, metrics
        from sklearn.model_selection import train_test_split, KFold, GridSearchCV
        from sklearn.ensemble import GradientBoostingRegressor
        # 設定要訓練的超參數組合
        n_estimators = [100, 200, 300]
        max_depth = [1, 3, 5]
        param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
        ## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
        grid_search = GridSearchCV(clf, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        # 開始搜尋最佳參數
        grid_result = grid_search.fit(x_train, y_train)
        # 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型
        # 印出最佳結果與最佳參數
        print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # 使用最佳參數重新建立模型
        clf_bestparam = GradientBoostingRegressor(
                            max_depth=grid_result.best_params_['max_depth'],
                            n_estimators=grid_result.best_params_['n_estimators'])
        # 訓練模型
        clf_bestparam.fit(x_train, y_train)
        # 預測測試集
        y_pred = clf_bestparam.predict(x_test)
        print(metrics.mean_squared_error(y_test, y_pred))
        ```
    * 延伸閱讀 :
        * [how to tune machine learning models](https://cambridgecoding.wordpress.com/2016/04/03/scanning-hyperspace-how-to-tune-machine-learning-models/)
        * [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* **Day_48 : Kaggle**
    * Kaggle 為全球資料科學競賽的網站，許多資料科學的競賽均會在此舉辦，吸引全球優秀的資料科學家參加
    * 主辦單位通常會把測試資料分為 public set 與 private set，參賽者上傳預測結果可以看到 public set 的成績，但比賽最終會使用 private set 的成績作為排名
    * Kernels 可以看見許多高手們分享的程式碼與結果，多半會以 jupyter notebook 呈現
    * Discussion 可以看到高手們互相討論可能的做法，或是資料中是否存在某些問題
    * [scikit-learn-practice](https://www.kaggle.com/c/data-science-london-scikit-learn)
* **Day_49 : 混和泛化 (Blending)**
    * **集成**是使用不同方式，結合多個或多種分類器，作為綜合預測的做法總稱
        * 將模型截長補短，可以說是機器學習的和議制/多數決
        * 其中分為資料層面的集成 : 如裝袋法(Bagging)、提升法(Boosting)
        * 以及模型的特徵集成 : 如混和泛化(Blending)、堆疊泛化(Stacking)
        * **裝袋法 (Bagging)** : 將資料放入袋中抽取，每回合結束後重新放回袋中重抽，在搭配弱分類器取平均或多數決結果，例如隨機森林
        * **提升法 (Boosting)** : 由之前模型的預測結果，去改便資料被抽到的權重或目標值
        * 將錯判的資料機率放大，正確的縮小，就是**自適應提升(AdaBoost, Adaptive Boosting)**
        * 如果是依照估計誤差的殘差項調整新目標值，則就是**梯度提升機 (Gradient Boosting Machine)** 的作法，只是梯度提升機還加上⽤用梯度來選擇決策樹分支
        * Bagging/Boosting : 使用不同資料、相同模型，多次估計的結果合成最終預測
        * Voting/Blending/Stacking : 使用同一自料不同模型，合成出不同預測結果
    * 混合泛化 (Blending)
        * 將不同模型的預測值加權合成，權重和為1如果取預測的平均 or 一人一票多數決(每個模型權重相同)，則又稱為投票泛化(Voting)
        * 容易使用且有效
        * 使用前提 : 個別**單模效果好**(有調教)並且**模型差異大**，單模要好尤其重要
        * 延伸閱讀 :
            * [Blending and Stacking](https://www.youtube.com/watch?v=mjUKsp0MvMI&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=27&t=0s)
            * [superblend](https://www.kaggle.com/tunguz/superblend/code)
            ```py
            new_train = pd.DataFrame([lr_y_prob, gdbt_y_prob, rf_y_prob])
            new_train = new_train.T

            clf = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=False, C=1.0)
            clf.fit(new_train, train_Y)

            coef = clf.coef_[0] / sum(clf.coef_[0]) 

            blending_pred = lr_pred*coef[0]  + gdbt_pred*coef[1] + rf_pred*coef[2]
            ```
* **Day_50 : 堆疊泛化 (Stacking)**
    * 不只將預測結果混和，而是使用預測結果**當新特徵**
    * Stacking 主要是把模型當作下一階的**特徵編碼器**來使用，但是**待編碼資料**與**訓練編碼器**的資料不可重複(訓練測試的不可重複性)
    * 若是將訓練資料切成兩份，待編碼資料太少，下一層的資料筆數就會太少，訓練編碼器的資料太少，則編碼器的強度就會不夠，因此使用 K-fold 拆分資料訓練，這樣資料沒有變少，編碼器也夠強韌，但 K 值越大訓練時間越長
    * 自我遞迴的 Stacking :
        * Q1：能不能新舊特徵一起用，再用模型預測呢?
            * A1：可以，這裡其實有個有趣的思考，也就是 : 這樣不就可以一直一直無限增加特徵下去? 這樣後面的特徵還有意義嗎? 不會 Overfitting 嗎?...其實加太多次是會 Overfitting 的，必需謹慎切分 Fold 以及新增次數
        * Q2：新的特徵，能不能再搭配模型創特徵，第三層第四層...一直下去呢?
            * A2：可以，但是每多一層，模型會越複雜 : 因此泛化(⼜又稱為魯棒性)會做得更好，精準度也會下降，所以除非第一層的單模調得很好，否則兩三層就不需要繼續往下了
        * Q3：既然同層新特徵會 Overfitting，層數加深會增加泛化，兩者同時用是不是就能把缺點互相抵銷呢?
            * A3：可以!!而且這正是 Stacking 最有趣的地方，但真正實踐時，程式複雜，運算時間又要再往上一個量級，之前曾有大神寫過 StackNet 實現這個想法，用JVM 加速運算，但實際上使用時調參困難，後繼使用的人就少了
        * Q4 : 實際上寫 Stacking 有這麼困難嗎?
            * 其實不難，就像 sklearn 幫我們寫好了許多機器學習模型，**mlxtend** 也已經幫我們寫好了 Stacking 的模型，所以用就可以了 (參考今日範例或 mlxtrend 官網)
        * Q5 : Stacking 結果分數真的比較高嗎?
            * 不一定，有時候單模更高，有時候 Blending 效果就不錯，視資料狀況而定
        * Q6 : Stacking 可以做參數調整嗎?
            * 可以，請參考 mlxtrend 的[調參範例](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/)，主要差異是參數名稱寫法稍有不同
        * Q7 : 還有其他做 Stacking 時需要注意的事項嗎?
            * 「分類問題」的 Stacking 要注意兩件事：記得加上 use_probas=True(輸出特徵才會是機率值)，以及輸出的總特徵數會是：模型數量*分類數量(回歸問題特徵數=模型數量量)
            ```py
            from mlxtend.classifier import StackingClassifier

            meta_estimator = GradientBoostingClassifier(tol=100, subsample=0.70, n_estimators=50, 
                                                    max_features='sqrt', max_depth=4, learning_rate=0.3)
            stacking = StackingClassifier(classifiers =[lr, gdbt, rf], use_probas=True, meta_classifier=meta_estimator)
            stacking.fit(train_X, train_Y)
            stacking_pred = stacking.predict(test_X)
            ```
* **Day_51~53 : Kaggle期中考**
    * [Enron Fraud Dataset 安隆公司詐欺案資料集](https://www.kaggle.com/c/ml100)
        * 如何處理存在各種缺陷的真實資料
        * 使用 val / test data 來了解機器學習模型的訓練情形
        * 使用適當的評估函數了解預測結果
        * 應用適當的特徵工程提升模型的準確率
        * 調整機器學習模型的超參數來提升準確率
        * 清楚的說明文件讓別人了解你的成果




