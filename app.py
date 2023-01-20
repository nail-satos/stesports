"""
Streamlitによる視線解析AIシステムの開発
"""

### 必要な標準ライブラリをインポートする
import copy             # ディープコピー
import os
import math             # nanの判定（math.isnan）
import json
from PIL import Image   # ロゴの表示用
import datetime

### 自作のモジュールをインポートする ###
# from modules.generic import func_html
import func_html


### 必要な外部ライブラリをインポートする
import streamlit as st
import streamlit.components.v1 as stc

import chardet          # 文字コードの判定
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 
from itertools import chain

# 正規化用のモジュールをインポート
from sklearn import preprocessing

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

#min-maxスケーリング
from sklearn.preprocessing import MinMaxScaler

#インスタンスの生成
minmax_sc = MinMaxScaler()

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


# アプリケーションの初期化をする関数
def init_app():

    # サイト全体の設定
    st.set_page_config(
        page_title="eスポーツ視線解析 ver0.1", 
        layout="wide", 
        initial_sidebar_state="auto")

    ### 各種フラグなどを初期化するセクション ###
    if 'init_flg' not in st.session_state:

        st.session_state['init_flg'] = True         # 初期化(済)フラグをオンに設定
        st.session_state['df_csv'] = []             # DataFrame形式のCSV
        st.session_state['check_flg'] = False       # csv初期検査の実行済フラグ
        st.session_state['df_train_tmp'] = []       # 訓練データの一時保管用（読み込んだ直後のデータのみ保持）



def st_display_table(df: pd.DataFrame):
    """
    Streamlitでデータフレームを表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム

    Returns
    -------
    なし
    """

    # データフレームを表示
    st.subheader(f'抜粋（{len(df)}件）')
    st.table(df)

    # 参考：Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_rtree(clf, features):
    """
    Streamlitでランダムフォレストの重要度を可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列群

    Returns
    -------
    なし
    """

    # 重要度の抽出
    feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    feature_importances = feature_importances.to_frame(name='重要度').sort_values(by='重要度', ascending=False)

    # TOP20可視化
    feature_importances[0:20].sort_values(by='重要度').plot.barh()
    plt.legend(loc='lower right')
    # plt.show()
    st.pyplot(plt)


def st_display_rtree(clf, features):
    """
    Streamlitでランダムフォレストの重要度を可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列群

    Returns
    -------
    なし
    """

    # 重要度の抽出
    feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    feature_importances = feature_importances.to_frame(name='重要度').sort_values(by='重要度', ascending=False)

    # TOP20可視化
    feature_importances[0:20].sort_values(by='重要度').plot.barh()
    plt.legend(loc='lower right')
    # plt.show()
    st.pyplot(plt)


def ml_drtree_pred(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm,
    depth: int,
    t_size: float) -> list:
    """
    決定木、またはランダムフォレストで学習と予測を行う関数
    
    Parameters
    ----------
    X : pd.DataFrame
        説明変数の列群
    y : pd.Series
        目的変数の列
    algorithm :
        'dtree' -> 決定木 または 'rtree' -> ランダムフォレスト
    depth : int
        決定木の深さ（ランダムフォレストの場合は無効）
    t_size: float
        訓練用データの割合（データ分割時）

    Returns
    -------
    list: [学習済みモデル, 訓練データでの予測値, 訓練データでの予測精度, 検証データでの予測値, 検証データでの予測精度]
    """

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=t_size, random_state=0, stratify=y)

    print(valid_x)

    # train_x = X
    # train_y = y
    
    # # データを水増し（オーバーサンプリング）する
    # oversample = SMOTE(sampling_strategy=0.5, random_state=0)
    # train_x, train_y = oversample.fit_resample(train_x, train_y)


    if algorithm == 'dtree':
        # 分類器の設定
        clf = DecisionTreeClassifier(max_depth=depth)

    elif algorithm == 'rtree':
        # 分類器の設定
        clf = RandomForestClassifier(random_state=0)

    # 学習
    clf.fit(train_x, train_y)

    # 戻り値の初期化
    train_scores = []
    # valid_scores = []

    # 訓練データで予測 ＆ 精度評価
    train_pred = clf.predict(train_x)
    
    # if np.count_nonzero(train_pred == '初級者') == 0:
    #     # 予測が全て'No'だった場合...
    #     train_scores = [0, 0, 0]
    #     valid_scores = [0, 0, 0]
    #     train_pred = np.nan
    #     valid_pred = np.nan
    #     print('no')

    # 目的変数を0,1に変換
    y_true = pd.get_dummies(train_y, drop_first=False)
    y_true = y_true['上級者'] # 型をSeriesに変換
    y_pred = pd.get_dummies(train_pred, drop_first=False)
    y_pred = y_pred['上級者'] # 型をSeriesに変換

    train_scores.append(round(accuracy_score(y_true, y_pred),3))
    train_scores.append(round(recall_score(y_true, y_pred),3))
    train_scores.append(round(precision_score(y_true, y_pred),3))

    # # # 検証データで予測 ＆ 精度評価
    # valid_pred = clf.predict(valid_x)

    # # 目的変数を0,1に変換
    # y_true = pd.get_dummies(valid_y, drop_first=False)
    # y_true = y_true['上級者'] # 型をSeriesに変換
    # y_pred = pd.get_dummies(valid_pred, drop_first=False)
    # y_pred = y_pred['上級者'] # 型をSeriesに変換

    # valid_scores.append(round(accuracy_score(y_true, y_pred),3))
    # valid_scores.append(round(recall_score(y_true, y_pred),3))
    # valid_scores.append(round(precision_score(y_true, y_pred),3))

    # return [clf, train_pred, train_scores, valid_pred, valid_scores]
    return [clf, train_pred, train_scores]



# データフレームの先頭と末尾をper(0.01～0.45)だけカットしてインデックスを再付番する関数
def cut_head_tail(df, per = 0.1):

    df_tmp = copy.deepcopy(df)  # ディープコピー

    # 最初と最後のper%を除外する
    head_tail_count = int(len(df_tmp) * per)
    df_tmp = df_tmp.tail(len(df_tmp) - head_tail_count)
    df_tmp = df_tmp.head(len(df_tmp) - head_tail_count)

    df_tmp.index = np.arange(1, len(df_tmp)+1)  # インデックスを1から再付番

    return df_tmp


# 列名を付け直して不要な列を削除する関数
def set_columns(df):

    # 列名の付け直し
    df.columns = ['タイムスタンプ', 'drop1', 'drop2', 'drop3', 'まばたき', '視線座標_横', '視線座標_縦']
    df = df[['タイムスタンプ', 'まばたき', '視線座標_横', '視線座標_縦']]

    return df


# データフレームの最初と最後の行から経過時間を算出する関数
def calc_elapsed_sec(df):

    str_start = df.iat[ 0, 0]
    str_end   = df.iat[-1, 0]

    # str型からdate型に変換（2022/10/04 16:44:54　→　2022-10-04 16:44:54）
    date_start = datetime.datetime.strptime(str_start[0:19], '%Y/%m/%d %H:%M:%S')   # datetime.datetime型
    date_end   = datetime.datetime.strptime(str_end[0:19], '%Y/%m/%d %H:%M:%S')     # datetime.datetime型
    date_elapsed = date_end - date_start        # datetime.timedelta型
    elapsed_sec = int(date_elapsed.total_seconds())  # timedelta型 → float型 → int型（秒数）に変換

    return elapsed_sec


# データの前処理を行う関数
def preprocess_data(df):

    # 列名を付け直して不要な列を削除する
    df = set_columns(df)

    # データフレームの先頭と末尾を10%カットしてインデックスを再付番する
    df = cut_head_tail(df, 0.1)

    # はずれ値の除去
    thrx1 = df['視線座標_横'].quantile(0.98)
    thrx2 = df['視線座標_横'].quantile(0.02)

    thry1 = df['視線座標_縦'].quantile(0.98)
    thry2 = df['視線座標_縦'].quantile(0.02)

    df = df.query('@thrx2 < 視線座標_横 <= @thrx1')
    df = df.query('@thry2 < 視線座標_縦 <= @thry1')                

    # インデックスを再付番する
    df = cut_head_tail(df, 0.0)

    return df


# データフレームから
# 経過時間・表示件数・データの先頭部分を画面に表示する関数
def view_data(df):

    col = st.columns([5, 5]) # カラムを作成
    col[0].subheader(f'データ数')
    col[0].info(f'{len(df)} 件')


    # データフレームの最初と最後の行から経過時間を算出する関数を呼び出す
    elapsed_sec = calc_elapsed_sec(df)

    col[1].subheader(f'計測時間')
    col[1].info(f'{elapsed_sec} 秒')
    # st.session_state['elapsed_sec'] = elapsed_sec   # セッションステートに計測時間を保存

    st.caption('※注釈：データ数と計測時間は、全データの「最初の10%」と「最後の10%」を除外して算出しています')

    # スライダーの表示（表示件数）
    cnt = st.sidebar.slider('表示するデータ', 1, len(df), 1, 1)

    # テーブルの表示（10件だけスライス）
    start_cnt = cnt - 1
    st_display_table(df[start_cnt : start_cnt+10])    



# データフレームから
# 「視線移動量(横[0]・縦[1])(正規化_横[2]・正規化_縦[3])」
# 「ミニマップを見た割合[4]」「HPバーを見た割合[5]」
# 「戦闘ログを見た割合[6]」「武器弾薬を見た割合[7]」
# 「まばたきの間隔[8]・割合[9]」を解析する関数
def analyze_data(df):

    list_ret = []

    # データフレームから列を抽出する
    posi_x = df['視線座標_横']
    posi_y = df['視線座標_縦']

    ###************************************************************************###
    # 視線移動量を算出するセクション
    ###************************************************************************###

    # 一個前のデータとの差を計算
    posi_x1 = abs(np.diff(posi_x))
    posi_y1 = abs(np.diff(posi_y))

    # 移動量の平均を算出
    average_x = np.mean(posi_x1)
    average_y = np.mean(posi_y1)

    # 戻り値用のリストに追加
    list_ret.append(average_x)
    list_ret.append(average_y)

    ###************************************************************************###
    # 正規化を算出するセクション
    ###************************************************************************###

    # posi_xとposi_yを正規化して、新しい変数に格納
    posi_x = preprocessing.minmax_scale(posi_x)
    posi_y = preprocessing.minmax_scale(posi_y)

    # 一個前のデータとの差を計算
    posi_x1 = abs(np.diff(posi_x))
    posi_y1 = abs(np.diff(posi_y))

    # 移動量の平均を算出
    average_x = np.mean(posi_x1)
    average_y = np.mean(posi_y1)

    # 戻り値用のリストに追加
    list_ret.append(average_x)
    list_ret.append(average_y)

    #　カラム名を付けてdfに変換
    kekka_x = pd.DataFrame(data=posi_x, columns=['x_seikika'])
    kekka_y = pd.DataFrame(data=posi_y, columns=['y_seikika'])

    # print('\n***** 正規化させた全体のデータ *****')
    kekka = pd.concat([kekka_x, kekka_y], axis=1)

    # print('\n***** データ数 *****')
    data_kekka = len(kekka)


    ###************************************************************************###
    # ミニマップを見た割合を算出するセクション
    ###************************************************************************###

    # print('\n***** MAPを見ている時間 *****')
    map = kekka.query('x_seikika < 0.3 & y_seikika < 0.3')

    # print('\n***** データ数 *****')
    data_map = len(map)

    # print('\n***** 割合 *****')
    mapw = data_map/data_kekka

    # 戻り値用のリストに追加
    list_ret.append(mapw)


    ###************************************************************************###
    # 戦闘ログを見た割合を算出するセクション
    ###************************************************************************###

    kp = kekka.query('x_seikika > 0.7 & y_seikika < 0.3')

    # print('\n***** データ数 *****')
    data_kp = len(kp)

    # print('\n***** 割合 *****')
    kpw = data_kp/data_kekka

    # 戻り値用のリストに追加
    list_ret.append(kpw)


    ###************************************************************************###
    # HPバーを見た割合を算出するセクション
    ###************************************************************************###

    # print('\n***** HPを見ている時間 *****')
    hp = kekka.query('x_seikika < 0.3 & y_seikika > 0.7')

    # print('\n***** データ数 *****')
    data_hp = len(hp)

    # print('\n***** 割合 *****')
    hpw = data_hp/data_kekka

    # 戻り値用のリストに追加
    list_ret.append(hpw)

    print(hpw)


    ###************************************************************************###
    # 武器弾数を見た割合を算出するセクション
    ###************************************************************************###

    wp = kekka.query('x_seikika > 0.7 & y_seikika > 0.7')

    # print('\n***** データ数 *****')
    data_wp = len(wp)

    # print('\n***** 割合 *****')
    wpw = data_wp/data_kekka

    # 戻り値用のリストに追加
    list_ret.append(wpw)


    # ###************************************************************************###
    # # 画面の中央以外を見た割合を算出するセクション
    # ###************************************************************************###

    # # print('\n***** 画面の中央付近を見ている時間 *****')
    # center = kekka.query('x_seikika > 0.4 & x_seikika < 0.6 & y_seikika > 0.2 & y_seikika < 0.8')

    # # print('\n***** データ数 *****')
    # data_center = len(center)

    # # print('\n***** 割合 *****')
    # centerw = data_center/data_kekka

    # # 戻り値用のリストに追加
    # # list_ret.append(centerw)
    # list_ret.append(1 - centerw)


    ###************************************************************************###
    # まばたきの割合を算出するセクション
    ###************************************************************************###

    # まばたきの変化点から、まばたきの回数を求める
    df["まばたき変化点"]=df["まばたき"].diff().apply(lambda x:0 if x==0 else 1 )
    blink_change_count = df['まばたき変化点'].sum() / 2

    # まばたきの間隔を求める
    elapsed_sec = calc_elapsed_sec(df)
    blink_interval = elapsed_sec / blink_change_count

    # 目を閉じていた割合を求める
    blink_count = df['まばたき'].sum()
    blink_per = blink_count / data_kekka

    # 戻り値用のリストに追加
    list_ret.append(blink_interval)
    list_ret.append(blink_per)


    return list_ret


def view_analyze_data(df):

    col = st.columns([5, 5]) # カラムを作成

    col[0].subheader(f'データ数')
    col[0].info(f'{len(df)} 件')

    # データフレームの最初と最後の行から経過時間を算出する関数を呼び出す
    elapsed_sec = calc_elapsed_sec(df)

    col[1].subheader(f'計測時間')
    col[1].info(f'{elapsed_sec} 秒')

    # データフレームから「視線移動量(横・縦)」「ミニマップを見た割合」「HPバーを見た割合」「まばたきの割合」を解析する関数
    lst = analyze_data(df)

    col[0].subheader(f'視線移動量(横方向)')
    col[1].subheader(f'視線移動量(縦方向)')

    # 視線移動量（生データ）を表示
    col[0].success(f'{int(lst[2]*10000)}')
    col[1].success(f'{int(lst[3]*10000)}')

    # 視線移動量（正規化）を表示
    col[0].caption(f'正規化前の値（{lst[0]}）')
    col[1].caption(f'正規化前の値（{lst[1]}）')

    # 割合データを表示
    col[0].subheader(f'ミニマップを見た割合')
    col[0].warning(f'{lst[4] * 100:.02f} ％')

    # 割合データを表示
    col[1].subheader(f'戦闘ログを見た割合')
    col[1].warning(f'{lst[5] * 100:.02f} ％')

    # 割合データを表示
    col[0].subheader(f'HPバーを見た割合')
    col[0].warning(f'{lst[6] * 100:.02f} ％')

    # 割合データを表示
    col[1].subheader(f'武器弾薬を見た割合')
    col[1].warning(f'{lst[7] * 100:.02f} ％')

    # まばたきデータを表示
    col[0].subheader(f'まばたきの間隔')
    col[0].error(f'{lst[8]:.02f} 秒に1回')
    col[1].subheader(f'目を閉じていた割合')
    col[1].error(f'{lst[9] * 100:.02f} ％')


# def edit_analyze_data(df):

#     col = st.columns([5, 5]) # カラムを作成

#     col[0].subheader(f'データ数')
#     col[0].info(f'{len(df)} 件')

#     # データフレームの最初と最後の行から経過時間を算出する関数を呼び出す
#     elapsed_sec = calc_elapsed_sec(df)

#     col[1].subheader(f'計測時間')
#     col[1].info(f'{elapsed_sec} 秒')

#     # データフレームから「視線移動量(横・縦)」「ミニマップを見た割合」「HPバーを見た割合」「まばたきの割合」を解析する関数
#     lst = analyze_data(df)

#     # col[0].subheader(f'視線移動量(横方向)')
#     # col[1].subheader(f'視線移動量(縦方向)')

#     # 視線移動量（生データ）を表示
#     col[0].number_input(label='視線移動量(横方向)', value = int(lst[2]*10000))
#     col[1].number_input(label='視線移動量(縦方向)', value = int(lst[3]*10000))

#     # 割合データを表示
#     col[0].number_input(label='ミニマップを見た割合(％)', value = lst[4] * 100)
#     col[1].number_input(label='戦闘ログを見た割合(％)', value = lst[5] * 100)
#     col[0].number_input(label='HPバーを見た割合(％)', value = lst[6] * 100)
#     col[1].number_input(label='武器弾薬を見た割合(％)', value = lst[7] * 100)



# CSVファイルからプレイヤーレベルと列名を指定して平均値を取得する関数
def get_train_mean(file_name, level_name, col_name):

    df_train = pd.read_csv(file_name, encoding="utf_8_sig" ,index_col=0)
    df_beginner = df_train.query('プレイヤーレベル == @level_name')

    return df_beginner[col_name].mean()


def view_bar_graph(your_data, scale, col_name, label):

    # CSVファイルからプレイヤーレベルと列名を指定して平均値を取得する
    mean_beginner = get_train_mean('train.csv', '初級者', col_name)
    mean_senior   = get_train_mean('train.csv', '上級者', col_name)

    lst = [your_data * scale, mean_beginner * scale, mean_senior * scale]

    df_graph = pd.DataFrame({'プレイヤーレベル':['あなた','初級者','上級者'], label : lst})

    fig, ax = plt.subplots()
    plt.title(label, fontsize=20)   # タイトル
    plt.grid(True)                  # 目盛線の表
    sns.barplot(data=df_graph, x="プレイヤーレベル", y=label, ax=ax)

    min_value = min(lst)
    max_value = max(lst)
    plt.ylim(min_value * 0.90, max_value * 1.10)  # 最小値と最大値の設定
    plt.ylabel('')
    st.pyplot(fig)


# 2枚の画像を重ね合わせる関数
def view_overlap_image(img1, img2, caption_text):

        # 画像を16：9にする(APEX)
        img = Image.open("APEX.jpg")
        img_resize = img.resize((1600, 900))
        img_resize.save("APEX_resize.jpg")

        #　グラフの画像のヒートマップ部分のみトリミング
        im = Image.open('GRAPH.jpg')
        im_crop = im.crop((29, 117, 408, 504))
        im_crop.save('GRAPH_crop.jpg', quality=95)

        # 画像を16：9にする
        img = Image.open("GRAPH_crop.jpg")
        img_resize = img.resize((1600, 900))
        img_resize.save("GRAPH_resize.jpg")

        #グラフの画像を透けさせる
        im1 = Image.open("GRAPH_resize.jpg")
        im1.putalpha(180)
        im2 = Image.open("APEX_resize.jpg")

        #画像を合成させ出力
        im2.paste(im1, (0, 0), im1)
        im2.save("out.jpg")

        st.image(im2, caption='プレイ画面との重ね合わせ',use_column_width=True)    
        st.image(img_resize, caption=caption_text, use_column_width=True) 


# アプリケーションのレイアウトを構築する関数
def view_screen():

    # サイドメニューの設定
    usr_menus = ['【選択してください】', 'データの登録', 'データの確認', 'データの分析（全体）', 'データの分析（視線移動量）', 'データの分析（ミニマップ）', 'データの分析（戦闘ログ）', 'データの分析（ＨＰバー）', 'データの分析（武器弾薬）', 'データの分析（まばたき）', '視点のヒートマップ表示', 'プレイヤーレベルの診断']
    usr_choice = st.sidebar.selectbox("メニューを選択してください", usr_menus)

    dev_menus = ['《開発者用のメニュー》', '訓練データの登録', '解析データの作成', '解析データの出力', '解析モデルの訓練', ]
    dev_choice = st.sidebar.selectbox("開発者用メニュー", dev_menus, disabled =True)


    ###************************************************************************###
    # 開発者用セクション《訓練データの登録》
    ###************************************************************************###
    if dev_menus.index(dev_choice) == 1:
        
        # ファイルアップローダー
        uploaded_file = st.sidebar.file_uploader('CSVをドラッグ＆ドロップしてください', type=['csv'])
        
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 文字コードの判定
            stringio = uploaded_file.getvalue()
            enc = chardet.detect(stringio)['encoding']

            # データフレームの読み込み
            df_load = pd.read_csv(uploaded_file, encoding=enc) 
            
            # データの前処理を行う関数を呼び出す
            df_tmp = preprocess_data(df_load)

            # データフレームから経過時間・表示件数・データの先頭部分を画面に表示する関数
            view_data(df_tmp)

            # データフレームをセッションステートに保存
            st.session_state['df_train_tmp'] = df_tmp

        # if st.sidebar.button('訓練データの削除'):

        #     # セッションステートから訓練データを削除
        #     del st.session_state['df_train']

        #     # 確認用のボタンを表示
        #     st.sidebar.button('再読込')

        # 開発者用セクションの番兵
        st.stop()


    ###************************************************************************###
    # 開発者用セクション《解析データの作成》
    ###************************************************************************###
    if dev_menus.index(dev_choice) == 2:


        # 訓練データを読み込み済みの場合...
        if 'df_train_tmp' in st.session_state:

            # ページヘッダの表示
            st.header(dev_choice)

            df = st.session_state['df_train_tmp']     # セッションステートから戻す

            # BACH_SIZE = 5000    # バッチサイズ(1ブロックのデータ数)を設定
            BLOCK_SIZE = 5000    # 1ブロックのデータ数を設定

            if len(df) >= BLOCK_SIZE:

                # ブロック数を算出
                block_count = int(len(df) / BLOCK_SIZE)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('訓練データのブロック(小分け)', 1, block_count)

                # スライスの始点と終点を算出
                start_position = (cnt - 1) * BLOCK_SIZE + 1
                end_position = start_position + BLOCK_SIZE - 1

                # 5000件ずつスライスしたデータフレームを作成
                df_slice = df.loc[start_position:end_position, :]

                # # データフレームから経過時間・表示件数・データの先頭部分を画面に表示する関数
                # view_data(df_slice)

                # 解析した結果を2カラムで表示する関数
                view_analyze_data(df_slice)

                player_levels = ['《選択してください》','初級者', '上級者']
                level_choice = st.sidebar.selectbox("このデータのプレイヤーレベル", player_levels)

                # 解析データの追加ボタン
                add_button = st.sidebar.button('解析データを追加する')
                if add_button:

                    if player_levels.index(level_choice) != 0:    # プレイヤーレベルが選択されているか

                        # ブロックごとにループを回す
                        for cnt in range(1, block_count + 1):

                            # スライスの始点と終点を算出
                            start_position = (cnt - 1) * BLOCK_SIZE + 1
                            end_position = start_position + BLOCK_SIZE - 1

                            # 5000件ずつスライスしたデータフレームを作成
                            df_slice = df.loc[start_position:end_position, :]

                            # 視線データを解析する関数
                            lst = analyze_data(df_slice)
                            # lst.append(level_choice)    # プレイヤーレベルをリストの末尾に追加
                            lst.insert(0, level_choice)     # プレイヤーレベルをリストの先頭に追加

                            if 'lst_train_pool' not in st.session_state:
                                # 解析データがセッションステートに存在しない場合...
                                st.session_state['lst_train_pool'] = []         # 解析データのプールを初期化

                            # 解析データのプールに追加
                            tmp_lst = st.session_state['lst_train_pool']
                            tmp_lst.append(lst)
                            st.session_state['lst_train_pool'] = tmp_lst

                            print(st.session_state['lst_train_pool'])
                    
                    else:
                        st.sidebar.warning('プレイヤーレベルを選択してください')


            else:
                st.warning('訓練データの件数が規定に達していません')

            # # スライダーの表示（表示件数）
            # cnt = st.sidebar.slider('表示する件数', 1, len(df), 1, 1)

        else:
            st.warning('訓練用データを登録してください')
            


    ###************************************************************************###
    # 開発者用セクション《解析データの出力》
    ###************************************************************************###
    if dev_menus.index(dev_choice) == 3:

        # 解析データを作成済みの場合...
        if 'lst_train_pool' in st.session_state:

            lst = st.session_state['lst_train_pool']     # セッションステートから戻す
            df = pd.DataFrame(lst)                       # リストをデータフレームに変換
            df.columns = ['プレイヤーレベル', '視線移動量_生横', '視線移動量_生縦', '視線移動量_正横', '視線移動量_正縦', 'ミニマップ割合', '戦闘ログ割合', 'ＨＰバー割合', '武器弾薬割合', 'まばたき間隔', 'まばたき割合']

            if len(df) >= 2: 

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示するデータ', 1, len(df), 1, 1)

                # テーブルの表示（10件だけスライス）
                start_cnt = cnt - 1
                st_display_table(df[start_cnt : start_cnt+10])    
            else:
                st_display_table(df)    

            export_file_name = st.sidebar.text_input('出力ファイル名', placeholder='例：train（拡張子は不要）')
            export_button = st.sidebar.button('解析データをCSV形式で出力する')

            if export_button:
                if not export_file_name:
                    st.sidebar.warning('CSVファイル名を入力してください')
                else:
                    csv_string = df.to_csv().encode('utf-8_sig')                    

                    st.sidebar.download_button(
                        label="CSVファイルをダウンロードする",
                        file_name=str(export_file_name)+'.csv',
                        mime='text/csv',
                        data=csv_string,
                    )

            # 解析データの削除ボタン
            init_button = st.sidebar.button('解析データを削除する')
            if init_button:
                if 'lst_train_pool' in st.session_state:
                    del st.session_state['lst_train_pool']         # 解析データのプールを初期化



    # 開発者用セクションの番兵
    if dev_menus.index(dev_choice) != 0:

        # 処理をここでストップ
        st.stop()


    ###************************************************************************###
    # ユーザー用セクション
    ###************************************************************************###

    # タイトルを表示する
    temps = [0 , 1]
    if usr_menus.index(usr_choice) in temps:

        # メインレイアウトの設定
        st.title("NCC eスポーツ視線解析AI（APEX版）")
        st.write("開発協力：株式会社ガゾウ／千葉大学 大学院／新潟大学 工学部 工学科 協創経営プログラム／株式会社新潟人工知能研究所")


        st.image('title.png')


    # メニュー：《データの登録》
    if usr_menus.index(usr_choice) == 1:

        # ファイルアップローダー
        uploaded_file = st.sidebar.file_uploader('CSVをドラッグ＆ドロップしてください', type=['csv'])

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 文字コードの判定
            stringio = uploaded_file.getvalue()
            enc = chardet.detect(stringio)['encoding']

            # データフレームの読み込み
            df = pd.read_csv(uploaded_file, encoding=enc) 

            ###************************************************************************###
            # 読み込んだCSVファイルのエラーチェック（とりあえず簡易版）＆前処理セクション
            ###************************************************************************###

            error_flg = False   # エラーチェック用のフラグ

            # 列数チェック（正常値 = 7）
            if len(df.columns) != 7:
                error_flg = True

            if error_flg == True:
                st.error(f'CSVデータにエラーが見つかりました。修正して再アップロードしてください。')

            else:
                ###************************************************************************###
                # 前処理のセクション
                ###************************************************************************###

                # データの前処理を行う関数を呼び出す
                df_tmp = preprocess_data(df)

                st.sidebar.success(f'視線データが正常に読み込めました\n\n《データの確認》に進んでください')
                st.session_state['check_flg'] = True

                st.session_state['df_csv'] = df_tmp # データフレームをセッションステートに保存


    # メニュー：《データの確認》
    if usr_menus.index(usr_choice) == 2:

        if st.session_state['check_flg'] == True:

            df = st.session_state['df_csv']     # セッションステートから戻す

            # ページヘッダの表示
            st.header(usr_choice)

            # データフレームから
            # 経過時間・表示件数・データの先頭部分を画面に表示する関数
            view_data(df)

            # サイドバーのメッセージ
            st.sidebar.success('データのプレビューを表示しました\n\n《データの分析》に進んでください')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（全体）》
    if usr_menus.index(usr_choice) == 3:

        if st.session_state['check_flg'] == True:

             # セッションステートから戻す
            df = st.session_state['df_csv']

            # ページヘッダの表示
            st.header(usr_choice)

            # 解析した結果を2カラムで表示する関数
            view_analyze_data(df)

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')

    # メニュー：《データの分析（視線移動量）》
    if usr_menus.index(usr_choice) == 4:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('<R>視線移動量</>はプレイ中にどれだけ視線が横や縦に動いたかを表す値です。<P>初級者</>と<G>上級者</>では、横方向の視線移動に大きな違いはありませんが、<R>縦方向</>は<G>上級者</>のほうが<R>移動量が大きい</>傾向があります。'))
            stc.html(text_html, height=175)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームから
            # 「視線移動量(横[0]・縦[1])(正規化_横[2]・正規化_縦[3])」
            # 「ミニマップを見た割合[4]」「HPバーを見た割合[5]」
            # 「戦闘ログを見た割合[6]」「武器弾薬を見た割合[7]」
            # 「まばたきの間隔[8]・割合[9]」を解析する関数
            lst = analyze_data(df)

            view_bar_graph(lst[2], 10000, '視線移動量_正横', '視線移動量(横方向)')
            view_bar_graph(lst[3], 10000, '視線移動量_正縦', '視線移動量(縦方向)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（ミニマップ）》
    if usr_menus.index(usr_choice) == 5:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_020.png', func_html.trans_html_tag('ゲーム中に<R>画面左上のミニマップ</>を見ていた割合です。<P>初級者</>よりも<G>上級者</>のほうが、よくミニマップを確認しています。'))
            stc.html(text_html, height=150)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームを解析する関数
            lst = analyze_data(df)
            view_bar_graph(lst[4], 100, 'ミニマップ割合', 'ミニマップを見た割合(％)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（戦闘ログ）》
    if usr_menus.index(usr_choice) == 6:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('ゲーム中に<R>画面右上の戦闘ログ</>を見ていた割合です。わずかな違いですが<P>初級者</>より<G>上級者</>のほうがチェックしているようです。'))
            stc.html(text_html, height=150)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームを解析する関数
            lst = analyze_data(df)
            view_bar_graph(lst[6], 100, '戦闘ログ割合', '戦闘ログを見た割合(％)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（ＨＰバー）》
    if usr_menus.index(usr_choice) == 7:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('ゲーム中に<R>画面左下のＨＰバー</>を見ていた割合です。<G>上級者</>のほうがＨＰバーを見る頻度が多く、<R>体力管理をしっかり行っている</>ことが読み取れます'))
            stc.html(text_html, height=150)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームを解析する関数
            lst = analyze_data(df)
            view_bar_graph(lst[7], 100, 'ＨＰバー割合', 'ＨＰバーを見た割合(％)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（武器弾薬）》
    if usr_menus.index(usr_choice) == 8:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_020.png', func_html.trans_html_tag('ゲーム中に<R>画面右下の武器弾薬</>を見ていた割合です。ゲームに慣れていない<P>初級者</>のほうが<R>武器弾薬を頻繁に確認してしまっている</>ようです。'))
            stc.html(text_html, height=150)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームを解析する関数
            lst = analyze_data(df)
            view_bar_graph(lst[7], 100, '武器弾薬割合', '武器弾薬を見た割合(％)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《データの分析（まばたき）》
    if usr_menus.index(usr_choice) == 9:

        if st.session_state['check_flg'] == True:

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('<G>上級者</>のほうが、ゲーム中に<R>まばたきの間隔が短く</>、<R>目を閉じている時間も長い</>ことが分かります。まばたきとゲームにどんな関係があるのかは現在、研究中です。'))
            stc.html(text_html, height=150)

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # データフレームを解析する関数
            lst = analyze_data(df)
            view_bar_graph(lst[8], 1, 'まばたき間隔', 'まばたきの間隔(秒)')
            view_bar_graph(lst[9], 1, 'まばたき割合', '目を閉じていた割合(％)')

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《視点のヒートマップ表示》
    if usr_menus.index(usr_choice) == 10:

        if st.session_state['check_flg'] == True:

            # サイドメニューの設定            
            map_menus = ['プレイ画面', 'ヒートマップ', 'ヘックス', '等高線風' ]
            map_choice = st.sidebar.selectbox("表示種別", map_menus, disabled =False)

            # メニュー：《視点のヒートマップ表示》
            if map_menus.index(map_choice) == 0:

                    # 画像の読み込み
                    image = Image.open('APEX.jpg')
                    st.image(image, caption='プレイ画面',use_column_width=True)

            # メニュー：《視点のヒートマップ表示》
            if map_menus.index(map_choice) != 0:

                # セッションステートから戻す
                df = st.session_state['df_csv']

                # データフレームから列を抽出する
                posi_x = df['視線座標_横']
                posi_y = df['視線座標_縦']

                # posi_xとposi_yを正規化して、新しい変数に格納
                posi_x = preprocessing.minmax_scale(posi_x)
                posi_y = preprocessing.minmax_scale(posi_y)

                #　カラム名を付けてdfに変換
                kekka_x = pd.DataFrame(data=posi_x, columns=['x_seikika'])
                kekka_y = pd.DataFrame(data=posi_y, columns=['y_seikika'])

                # yの値を反転（y = 1-y)
                kekka_y["y_seikika"] = 1 - kekka_y["y_seikika"]

                # print('\n***** 正規化させた全体のデータ *****')
                kekka = pd.concat([kekka_x, kekka_y], axis=1)

                # メニュー：《視点のヒートマップ表示》
                if map_menus.index(map_choice) == 1:

                    # ヒストグラム
                    fig = sns.jointplot(x="x_seikika", y="y_seikika", data=kekka, kind="hist", color="C4", bins=15)                           
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.axis("off")
                    # st.pyplot(fig)

                    # グラフを画像として保存
                    fig.savefig('GRAPH.jpg')

                    # 2枚の画像を重ね合わせる関数を呼び出す
                    view_overlap_image('APEX.jpg', 'GRAPH.jpg', map_choice)

                # メニュー：《視点のヒートマップ表示》
                if map_menus.index(map_choice) == 2:

                    # #六角形
                    fig = sns.jointplot(x="x_seikika", y="y_seikika", data=kekka, kind="hex", bins=15)
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.axis("off")
                    # st.pyplot(fig)

                    # グラフを画像として保存
                    fig.savefig('GRAPH.jpg')

                    # 2枚の画像を重ね合わせる関数を呼び出す
                    view_overlap_image('APEX.jpg', 'GRAPH.jpg', map_choice)

                # メニュー：《視点の等高線表示》
                if map_menus.index(map_choice) == 3:

                    print(len(kekka))
                    # データフレームをスライスで5おきで間引き（ダウンサンプリング）
                    kekka = kekka[::20]  

                    # 波
                    fig = sns.jointplot(x="x_seikika", y="y_seikika", data=kekka, kind="kde", color="blue")
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.axis("off")
                    # st.pyplot(fig)

                    # グラフを画像として保存
                    fig.savefig('GRAPH.jpg')

                    # 2枚の画像を重ね合わせる関数を呼び出す
                    view_overlap_image('APEX.jpg', 'GRAPH.jpg', map_choice)

                # # しぃちゃんからのアドバイス
                # text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('しぃちゃんからのアドバイスがここに入る<C><B>〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇！</>'))
                # stc.html(text_html, height=150)

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')


    # メニュー：《プレイヤーレベルの診断》
    if usr_menus.index(usr_choice) == 11:

        if st.session_state['check_flg'] == True:

            # セッションステートから戻す
            df = st.session_state['df_csv']

            # ページヘッダの表示
            st.header(usr_choice)

            # 解析した結果を2カラムで表示する関数
            # edit_analyze_data(df)

            col = st.columns([5, 5]) # カラムを作成

            col[0].subheader(f'データ数')
            col[0].info(f'{len(df)} 件')

            # データフレームの最初と最後の行から経過時間を算出する関数を呼び出す
            elapsed_sec = calc_elapsed_sec(df)

            col[1].subheader(f'計測時間')
            col[1].info(f'{elapsed_sec} 秒')

            # データフレームから「視線移動量(横・縦)」「ミニマップを見た割合」「HPバーを見た割合」「まばたきの割合」を解析する関数
            lst = analyze_data(df)

            # col[0].subheader(f'視線移動量(横方向)')
            # col[1].subheader(f'視線移動量(縦方向)')

            # 視線移動量（生データ）を表示
            x1 = col[0].number_input(label='視線移動量(横方向)', value = int(lst[2]*10000))
            x2 = col[1].number_input(label='視線移動量(縦方向)', value = int(lst[3]*10000))

            # 割合データを表示
            x3 = col[0].number_input(label='ミニマップを見た割合(％)', value = lst[4] * 100)
            x4 = col[1].number_input(label='戦闘ログを見た割合(％)', value = lst[5] * 100)
            x5 = col[0].number_input(label='HPバーを見た割合(％)', value = lst[6] * 100)
            x6 = col[1].number_input(label='武器弾薬を見た割合(％)', value = lst[7] * 100)

            # 訓練データをファイルから読み込む
            df_train = pd.read_csv('train.csv', encoding="utf_8_sig" ,index_col=0)
            # df_beginner = df_train.query('プレイヤーレベル == @level_name')

            # 説明変数と目的変数の設定
            train_X = df_train.drop(["プレイヤーレベル", '視線移動量_生横', '視線移動量_生縦', 'まばたき間隔', 'まばたき割合'], axis=1)   # 退職列以外を説明変数にセット
            train_Y = df_train["プレイヤーレベル"]                # 退職列を目的変数にセット

            # st_display_table(train_X)


            # ランダムフォレストによる予測
            clf, train_pred, train_scores, = ml_drtree_pred(train_X, train_Y, 'rtree', 0, 2/3)

            # # 検証データで予測 ＆ 精度評価
            list_x = [[x1/10000, x2/10000, x3/100, x4/100, x5/100, x6/100]]
            valid_X = pd.DataFrame(data=list_x, columns=train_X.columns)

            valid_pred = clf.predict(valid_X)
            valid_pred_proba = clf.predict_proba(valid_X)

            # st.info(valid_pred)
            # st.success(f'このデータは「{int(valid_pred_proba[0][0]*100)}%」の確率で「上級者の視線」です')
            st.markdown(f'# このデータは… :red[「{int(valid_pred_proba[0][0]*100)}%」]の確率で:green[上級者の視線]です')

            # 特徴量の設定（重要度の可視化用）
            features = ['視線移動量(横)', '視線移動量(縦)', 'ミニマップを見た割合', '戦闘ログを見た割合', 'ＨＰバーを見た割合', '武器弾薬を見た割合',]

            print(features)

            # しぃちゃんからのアドバイス
            text_html = func_html.make_html_balloon('enushi_010.png', func_html.trans_html_tag('入力されたデータをもとに<G>上級者の視線</>である確率を<R>AIが算出</>しました。<C>また確率を算出する際に重要視した要素を表したのが以下のグラフです。プレイ中は<R>HPバーやミニマップ</>で状況を把握しつつ、<B>武器弾薬を見ない</>ようになれると、<G>上級者</>に近づけるかも…しれません。'))
            stc.html(text_html, height=200)

            # 重要度の可視化
            st.caption('視線レベルの判定に使用したパラメーターの重要度（重み）')
            st_display_rtree(clf, features)

            st.balloons()

        else:
            st.warning('《データの登録》から、視線データをアップロードしてください')



def main():
    """
    メインモジュール
    """

    # アプリケーションの初期化
    init_app()

    # アプリケーションのレイアウトを構築    
    view_screen()


if __name__ == "__main__":
    main()

