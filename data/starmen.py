
# # ## starmen resave the demo
# # # resave images after augmentation
# # dataset_input_path = './data/'
# # starmen_df = os.path.join(dataset_input_path, "starmen", "output_random", "df.csv")
# # df = pd.read_csv(os.path.join(starmen_df), index_col=[0])
# # starmenaug_df = os.path.join(dataset_input_path, "starmen-augmentation", "output_random", "df.csv")
# # df = pd.read_csv(os.path.join(starmenaug_df), index_col=[0])
# timepoint = np.empty((0)) # 4000 # 1000 # 5000
# unqid = np.unique(df.id)
# for id in unqid:
#     indices = np.array(df[df.id == id].index)
#     tsort = np.array(np.argsort(df.t[indices]))
#     timepoint = np.concatenate((timepoint,tsort),0)
#
# df['timepoint'] = timepoint
# df.path = df.path.str.replace('/home/heejong/HDD4T/projects/pairwise-comparison-longitudinal/'
#                               'baseline/longitudinal_autoencoder/data/starmen/'
#                               'output_random/','')
# df.to_csv(starmen_df)
#
# df.path = df.path.str.replace('/home/heejong/HDD4T/projects/pairwise-comparison-longitudinal/'
#                               'baseline/longitudinal_autoencoder/data/starmen-augmentation/'
#                               'output_random/','')
# df.to_csv(starmenaug_df)
