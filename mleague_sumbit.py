import pandas as pd
from pyparsing import autoname_elements
from sklearn.decomposition import PCA
import seaborn as sns

import matplotlib
from matplotlib.font_manager import FontProperties
font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = font_prop.get_name()
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('mleague2021.csv', index_col=0, sep=',')
    print('df')
    print(df)

    df.drop(['総局数'], axis=1, inplace=True)
    df.drop(['1位'], axis=1, inplace=True)
    df.drop(['2位'], axis=1, inplace=True)
    df.drop(['3位'], axis=1, inplace=True)
    df.drop(['4位'], axis=1, inplace=True)


    label = df['Team']
    df.drop(['Team'], axis=1, inplace=True)

    x = df.iloc[:, 1:]  # 目的変数の列は0列目
    autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング
    print(autoscaled_x)

    pca = PCA()  # PCA を宣言してから
    pca.fit(autoscaled_x)  # PCA 実行

    score = pd.DataFrame(pca.transform(autoscaled_x), index=autoscaled_x.index)
    score['Team'] = label

    print('score')
    print(score)
    score.to_csv('mleague_pca.csv', encoding='utf-8')

    loadings = pd.DataFrame(pca.components_.T, index=autoscaled_x.columns)
    print('loadings')
    print(loadings)
    loadings.to_csv('mleague_loadings.csv', encoding='utf-8')

    contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)
    print('contribution_ratios')
    print(contribution_ratios)
    contribution_ratios.to_csv('mleague_contribution_ratios.csv', encoding='utf-8')

    groups = score.groupby('Team')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group[0], group[1], marker='o', linestyle='', ms=6, label=name)
    ax.legend()
    plt.savefig("mleague_scatter.png")
    plt.show()

    if 0:
        # 相関行列
        correlation_coefficients = df.corr()  # 相関行列の計算
        # 相関行列のヒートマップ 
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(12, 8))  # 画像サイズ
        sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=True, xticklabels=1, yticklabels=1)
        plt.xlim([0, correlation_coefficients.shape[0]])
        plt.savefig("mleague_heatmap.png")
        plt.show()


if __name__ == '__main__':
    main()
