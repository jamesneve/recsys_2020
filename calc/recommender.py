from scipy import stats

class Recommender:
    def __init__(self, user_data_df, likes_df):
        user_data_df = user_data_df.set_index('user_id')
        likes_df = likes_df.groupby('user_id')['partner_id'].apply(list)

        self.user_data_df = user_data_df
        self.likes_df = likes_df

    def get_liked_users(self, user_id):
        if user_id not in self.likes_df.index:
            return []
        liked_ids = self.likes_df[user_id]
        user_data_rows = self.user_data_df[self.user_data_df.index.isin(liked_ids)]

        return user_data_rows

    def get_user_data(self, user_id):
        if user_id in self.user_data_df.index:
            user_data = self.user_data_df[self.user_data_df.index == user_id]
            return user_data.to_dict('records')[0]
        return {}

    @staticmethod
    def aggregate_preference_scores(m_f_score, f_m_score):
        aggregated_score = stats.hmean([m_f_score, f_m_score])

        return aggregated_score
