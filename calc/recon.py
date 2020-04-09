import numpy as np

from .recommender import Recommender

# RECON as in https://dl.acm.org/doi/pdf/10.1145/1864708.1864747
# Using modification suggested in https://dl.acm.org/doi/pdf/10.1145/2808797.2809282
#      - Continuous attributes are not binned, but calculated by distance metric
class Recon(Recommender):
    preference_attributes = ['nationality_id', 'education_id', 'job_id', 'annual_income_id', 'horoscope_id',
                             'lodger_id', 'height', 'body_build_id', 'blood_type_id', 'marital_status_id', 'child_id',
                             'when_marry_id', 'want_child_id', 'smoking_id', 'drinking_id', 'how_to_meet_id',
                             'cost_of_date_id', 'nth_child_id', 'housework_id', 'relationship_status_id',
                             'sociability_id', 'purpose_id', 'age']
    continuous_attributes = ['height', 'age']

    def __init__(self, user_data_df, likes_df):
        Recommender.__init__(self, user_data_df, likes_df)
        self.preference_distributions = {}

    def calculate_reciprocal_preference(self, m_user_id, f_user_id):
        m_f_score = self.get_preference_score(m_user_id, f_user_id)
        f_m_score = self.get_preference_score(f_user_id, m_user_id)

        reciprocal_score = self.aggregate_preference_scores(m_f_score, f_m_score)

        return reciprocal_score

    def get_preference_score(self, user_x_id, user_y_id):
        user_x_prefs = self.get_preference_distribution(user_x_id)
        user_y_data = self.get_user_data(user_y_id)

        preference_score = self.calculate_score(user_x_prefs, user_y_data)
        return preference_score

    def calculate_score(self, user_x_prefs, user_y_data):
        preference_sum = 0.0
        if len(user_y_data) == 0 or len(user_x_prefs) == 0:
            return 0.5
        for attribute in self.preference_attributes:
            user_y_attribute = user_y_data[attribute]
            user_x_preference = user_x_prefs[attribute]
            if attribute in self.continuous_attributes:
                ca_preference = self.calculate_continuous_attribute_preference(user_y_attribute, user_x_preference)
                preference_sum += ca_preference
            else:
                if user_y_attribute in user_x_preference:
                    preference_sum += user_x_preference[user_y_attribute]

        overall_preference = preference_sum / float(len(self.preference_attributes))
        return overall_preference

    @staticmethod
    def calculate_continuous_attribute_preference(user_y_attribute, user_x_preference):
        ca_sum = 0.0
        ca_total = 0
        for key in user_x_preference:
            va = 100.0
            qa = (va - abs(user_y_attribute - key)) / va
            ca_sum += qa * user_x_preference[key]
            ca_total += user_x_preference[key]
        ca_preference = ca_sum / float(ca_total)
        return ca_preference

    def get_preference_distribution(self, user_id):
        if user_id in self.preference_distributions:
            return self.preference_distributions[user_id]
        else:
            liked_users = self.get_liked_users(user_id)
            preference_distributions = self.calculate_preference_distributions(liked_users)
            return preference_distributions

    def calculate_preference_distributions(self, liked_users):
        distributions = {}
        if len(liked_users) == 0:
            return distributions

        for attribute in self.preference_attributes:
            distributions[attribute] = {}

        density_increase = 1.0 / float(len(liked_users))
        density_increase = np.around(density_increase, decimals=3)

        for _, user in liked_users.iterrows():
            for attribute in self.preference_attributes:
                attribute_increase = density_increase
                if attribute in self.continuous_attributes:
                    attribute_increase = 1.0

                attribute_val = user[attribute]
                if attribute_val in distributions[attribute]:
                    distributions[attribute][attribute_val] += attribute_increase
                else:
                    distributions[attribute][attribute_val] = attribute_increase

        return distributions
