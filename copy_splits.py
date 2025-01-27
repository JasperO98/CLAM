import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Copying splits for whole slide classification')
parser.add_argument('--splits_csv', type=str)
parser.add_argument('--task', type=str, required=True, choices=[
    'task_1_tumor_vs_normal', 'task_2_tumor_subtyping',
    'desmel_s224_l0', 'desmel_s224_l1', 'desmel_s224_l2',
    'desmel_s512_l0', 'desmel_s512_l1', 'desmel_s512_l2',
])

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.isfile(args.splits_csv):
        print(args.splits_csv, 'does not exist')
        quit()
    
    split_dir = os.path.join('splits', 'desmel', str(args.task).replace('desmel_', ''))
    os.makedirs(split_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.splits_csv), usecols=['study_code', 'tile_dir', 'label', 'control_distmet_2024', 'fold1', 'fold2', 'fold3']).groupby('tile_dir', as_index=False).first()
    # Save the list of patients
    pd.DataFrame(
        df.groupby(['study_code', 'tile_dir', 'label']).first().index.tolist(),
        columns=['case_id', 'slide_id', 'label'],
    ).to_csv(os.path.join('dataset_csv', args.task + '.csv'), index=False)
    df.drop('study_code', axis=1, inplace=True)
    for fold in df.columns[3:]:
        n = int(fold.strip('fold')) - 1
        
        # dummies
        df_dummies = pd.get_dummies(df[['tile_dir', 'label', 'control_distmet_2024', fold]].set_index(['tile_dir', 'label', 'control_distmet_2024']), prefix='', prefix_sep='')[['train', 'val', 'test']]
        df_dummies.reset_index(['label', 'control_distmet_2024'], inplace=True)
        df_dummies['control_distmet_2024'] = df_dummies['control_distmet_2024'].fillna(df_dummies['label']).astype(int)
        df_dummies.drop('label', axis=1, inplace=True)
        df_dummies.rename(columns={'control_distmet_2024': 'label'}, inplace=True)
        df_dummies.set_index('label', append=True, inplace=True)
        
        # bool
        splits_bool = df_dummies.droplevel(axis=0, level='label').sort_values(['train', 'val', 'test'], ascending=False)
        splits_bool.index.name = None
        splits_bool.to_csv(os.path.join(split_dir, f'splits_{n}_bool.csv'))
        
        # descriptor
        splits_descriptor = df_dummies.groupby('label').sum()
        splits_descriptor.index.name = None
        splits_descriptor.to_csv(os.path.join(split_dir, f'splits_{n}_descriptor.csv'))
        
        # splits
        splits = pd.concat([
            pd.Series(df_dummies.loc[df_dummies['train']].index.get_level_values('tile_dir')),
            pd.Series(df_dummies.loc[df_dummies['val']].index.get_level_values('tile_dir')),
            pd.Series(df_dummies.loc[df_dummies['test']].index.get_level_values('tile_dir')),
        ], axis=1)
        splits.columns = ['train', 'val', 'test']
        splits.to_csv(os.path.join(split_dir, f'splits_{n}.csv'))
        
    pd.DataFrame(
        df_dummies.loc[df_dummies['test']].index.tolist(),
        columns=['slide_id', 'label'],
    ).to_csv(os.path.join('heatmaps', 'process_lists', args.task + '.csv'), index=False)
