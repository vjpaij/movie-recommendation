import pandas as pd
from fastai.collab import CollabDataLoaders, collab_learner

cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None, names=cols)
print(ratings_df.sample(5))

data = CollabDataLoaders.from_df(ratings_df, valid_pct=0.1)
print(data.valid_ds)
learn = collab_learner(data, n_factors=40, y_range=[0, 5.5], wd = 0.1)
learn.fit_one_cycle(5, 0.01)

#Looking at some predictions
# sample = next(iter(data.train_ds))
# print((sample))
(users, items), ratings = next(iter(data.valid_ds))
preds = learn.model(users, items)
print('User\tMovie\tReal\tPred\tDifference')
for p in list(zip(users, items, ratings, preds))[:16]:
    print('{}\t{:.1f}\t{:.1f}'.format(p[0], p[1], p[2], p[3], p[3]-p[2]))