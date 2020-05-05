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
        * 機器學習特徵是 **寧爛勿缺**，以前非樹狀模型為避免共線性，會希望類似特徵不要太多，但現在強力模型大多是樹狀模型，所以通通做成~~雞精~~特徵
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


