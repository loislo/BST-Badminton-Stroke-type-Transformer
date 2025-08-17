#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import os


def get_one_competition_result(folder_path: Path, first_A_is_top: bool):
    '''A is the winner and B is the loser.'''
    set_count = len(list(folder_path.glob('*.csv')))
    sets_collected_ls = []

    # For set1 and set2
    for i in range(1, 3):
        df = pd.read_csv(folder_path/f'set{i}.csv')
        df = df[['player', 'type']]

        if (first_A_is_top ^ (i == 2)):
            df['player'] = np.where(df['player'] == 'A', 'Top', 'Bottom')
        else:
            df['player'] = np.where(df['player'] == 'B', 'Top', 'Bottom')

        type_count = df.groupby(['player', 'type']).size()
        sets_collected_ls.append(type_count)

    # 處理 11 分換場
    if set_count == 3:
        df = pd.read_csv(folder_path/'set3.csv')
        df = df[['roundscore_A', 'roundscore_B', 'player', 'type', 'rally']]
        
        # Compute the split point
        i = df['roundscore_A'].searchsorted(11, side='left')
        if df.loc[i, 'roundscore_A'] <= df.loc[i, 'roundscore_B']:
            i = df['roundscore_B'].searchsorted(11, side='left')
        i_split = df['rally'].searchsorted(df.loc[i, 'rally'], side='right')
        df_1 = df.iloc[:i_split]
        df_2 = df.iloc[i_split:]

        df_1 = df_1[['player', 'type']]
        df_2 = df_2[['player', 'type']]

        if first_A_is_top:
            df_1['player'] = np.where(df_1['player'] == 'A', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'B', 'Top', 'Bottom')
        else:
            df_1['player'] = np.where(df_1['player'] == 'B', 'Top', 'Bottom')
            df_2['player'] = np.where(df_2['player'] == 'A', 'Top', 'Bottom')

        count_1 = df_1.groupby(['player', 'type']).size()
        count_2 = df_2.groupby(['player', 'type']).size()
        sets_collected_ls += [count_1, count_2]

    df = pd.concat(sets_collected_ls, axis=1)
    df = df.fillna(0)  # NaN 填 0 不然會出錯
    result = df.astype(int).sum(axis=1).sort_index()
    return result


def update_dataframes(df_dic: dict[str, pd.DataFrame], competition_name: str, result: pd.Series):
    '''Update dataframes.'''
    df_top = df_dic['Top Player']

    row = df_top[df_top['Video Name'] == competition_name].index
    for player in ['Top', 'Bottom']:
        cur_sheet = f'{player} Player'
        for stroke_name, stroke_count in result[player].items():
            col = np.argmax(df_dic[cur_sheet].columns == stroke_name)
            if col != 0:  # dataset 有未知球種
                df_dic[cur_sheet].iloc[row, col] = stroke_count
            else:
                raise Exception(f'球種 {stroke_name} 沒有統計到')


if __name__ == "__main__":
    match_csv_df = pd.read_csv('set/match.csv')[['video', 'downcourt']].set_index('video')
    match_ls = [p for p in Path("set").glob('*') if p.is_dir()]
    df_dic = pd.read_excel("class_total.xlsx", sheet_name=['Top Player', 'Bottom Player'])
    
    # 歸零
    df_dic['Top Player'].iloc[:, 2:] = 0
    df_dic['Bottom Player'].iloc[:, 2:] = 0
    
    for competition in match_ls:
        # if competition.name == 'CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage':
        #     os.system('pause')
        first_A_is_top = bool(match_csv_df.loc[competition.name, 'downcourt'])  # downcourt 是 1 代表 B 為 Bottom player
        result = get_one_competition_result(competition, first_A_is_top)
        # print(competition.name)
        # print(result.sum())
        update_dataframes(df_dic, competition.name, result)
    
    with pd.ExcelWriter('class_total_gen.xlsx') as writer:
        df_dic['Top Player'].to_excel(writer, sheet_name='Top Player', index=False)
        df_dic['Bottom Player'].to_excel(writer, sheet_name='Bottom Player', index=False)
