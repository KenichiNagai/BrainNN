#%%
from cProfile import label
from multiprocessing import dummy
from sklearn import datasets
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting 
import seaborn as sns


from IPython.display import display

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib
from matplotlib.font_manager import FontProperties
font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = font_prop.get_name()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import collections



def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)


def main():
    df = pd.read_csv('kankore3.csv', index_col=0, sep=',')

    df.dropna(subset=['艦名'], axis=0, inplace=True)
    df.drop(['備考'], axis=1, inplace=True)
    df.drop(['追加'], axis=1, inplace=True)
    df.drop(['速力'], axis=1, inplace=True)
    df.drop(['射程'], axis=1, inplace=True)
    df.drop(['艦名'], axis=1, inplace=True)
    df.drop(['弾薬'], axis=1, inplace=True)

    df.dropna(axis=0, inplace=True)
    df.replace("--", 0, inplace=True)

    label = df['艦種']
    df.drop(['艦種'], axis=1, inplace=True)

    # これしないと型エラーで計算がおかしくなる
    df = df.astype('float64')

    print(df.head())


    df.to_csv('kankore_pandas.csv', encoding='utf-8')
    # print(df.shape)

    
    # df.drop(['艦種'], axis=1, inplace=True)
    # dummy_df = df
    # dummy_df = pd.get_dummies(df, columns=['艦種'])
    # dataset = dummy_df


    dataset = df
    # 相関行列
    correlation_coefficients = dataset.corr()  # 相関行列の計算

    # # 相関行列のヒートマップ (相関係数の値あり) 
    # plt.rcParams['font.size'] = 12
    # plt.figure(figsize=(12, 8))  # この段階で画像のサイズを指定する
    # sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=True, xticklabels=1, yticklabels=1)
    # plt.xlim([0, correlation_coefficients.shape[0]])
    # plt.show()


    # # 特徴量の散布図
    # plt.rcParams['font.size'] = 8  # 横軸や縦軸の名前の文字などのフォントのサイズ
    # pd.plotting.scatter_matrix(dataset, c='blue', figsize=(10, 10)) # ここで画像サイズを指定
    # plt.show()

    word2id = collections.defaultdict(lambda: len(word2id) )
    sentence = label
    ids = [word2id[word] for word in sentence]
    # print("sentence    :", sentence)
    # print("id_sentence :", label )
    print("dict        :", dict(word2id) )
    dict_label = dict(word2id)


    print('dataset.iloc')
    x = dataset.iloc[:, 0:]  # 最初の列のbandgapを目的変数とする
    # x = dataset
    # print(x)
    # print(x - x.mean())
    # print(x.std())
    autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング
    print(autoscaled_x)

    pca = PCA()  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
    pca.fit(autoscaled_x)  # PCA を実行


    # https://qiita.com/oki_kosuke/items/43cb63134f9a03ebc79a

    # スコアとは各サンプルが各主成分軸上のどの座標に位置するかを表す値です。以下のコードでスコアを確認します。
    score = pd.DataFrame(pca.transform(autoscaled_x), index=autoscaled_x.index)
    score['艦種'] = label
    # score.replace(dict_label, inplace=True)


    print(score)
    score.to_csv('kankore_pca.csv', encoding='utf-8')

    loadings = pd.DataFrame(pca.components_.T, index=x.columns)
    print('loadings')
    print(loadings)

    contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)
    print('contribution_ratios')
    print(contribution_ratios)

    # # 主成分分析の結果を可視化します。第一主成分vs第二主成分平面に各サンプルをプロットします。また、各プロットはband_gap（目的変数）の値で色付けしておきます。
    # plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=dataset.iloc[:, -1], cmap=plt.get_cmap('jet'))
    # clb = plt.colorbar()
    # clb.set_label('艦種', labelpad=-20, y=1.1, rotation=0)
    # plt.xlabel('t1')
    # plt.ylabel('t2')
    # plt.show()

    groups = score.groupby('艦種')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group[0], group[1], marker='o', linestyle='', ms=6, label=name)
    ax.legend()
    plt.show()


    X = score[score['艦種'] == 0]
    print(X)
    y = X['No']
    # 決定境界を求める
    clf = LinearDiscriminantAnalysis(store_covariance=True)
    clf.fit(X, y)

    # どのような決定境界が引かれたかを確認する
    w = clf.coef_[0]
    wt = -1 / (w[1] / w[0])  ## wに垂直な傾きを求める
    xs = np.linspace(-10, 10, 100)
    ys_w = [(w[1] / w[0]) * xi for xi in xs]
    ys_wt = [wt * xi for xi in xs]

    fig = plt.figure(figsize=(7, 7))
    plt.title("決定境界の傾きを可視化", fontsize=20)
    plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=y)  # サンプルデータ
    plt.plot(xs, ys_w, "-.", color="k", alpha=0.5)  # ｗの向き
    plt.plot(xs, ys_wt, "--", color="k")  # ｗに垂直な向き

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

    # 求めたベクトルｗを元に１次元にデータを移した結果
    X_1d = clf.transform(X).reshape(1, -1)[0]
    fig = plt.figure(figsize=(7, 7))
    plt.title("１次元にデータを移した場合のデータの位置", fontsize=15)
    plt.scatter(X_1d, [0 for _ in range(len(X))], c=y)
    plt.show()





def main2():


    df = pd.read_csv('kankore2.csv', index_col=0, sep=',')

    df.dropna(subset=['艦名'], axis=0, inplace=True)
    df.drop(['備考'], axis=1, inplace=True)
    df.drop(['追加'], axis=1, inplace=True)
    df.drop(['速力'], axis=1, inplace=True)
    df.drop(['射程'], axis=1, inplace=True)

    df.dropna(axis=0, inplace=True)

    df.replace("--", 0, inplace=True)

    df.drop(['艦名'], axis=1, inplace=True)

    label = df['艦種']
    df.drop(['艦種'], axis=1, inplace=True)

    print(df.head())

    pca = PCA(n_components=5)
    df_pca = pd.DataFrame(pca.fit_transform(df))
    df_pca['label'] = label
    sns.scatterplot(data=df_pca, x=0, y=1, hue='label')
    plt.show()
    
    ev = pd.DataFrame(pca.explained_variance_ratio_)
    print(ev)
    



main()
# main2()
# %%
