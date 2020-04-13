from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .recommender import Recommender

from skimage.io import imread
from skimage.transform import resize

import numpy as np

import os

class ImRec(Recommender):

    IMAGES_FOLDER = "/Users/james.neve/Data/Preference/0409UserImages/Female/"
    IMAGE_DIM = 100
    M_F_MODEL_PATH = "/Users/james.neve/development/python/recsys_2020_recommender/models/face_data_siamese_200409_male.h5"
    F_M_MODEL_PATH = "/Users/james.neve/development/python/recsys_2020_recommender/models/face_data_siamese_200409_female.h5"

    def __init__(self, user_data_df, likes_df):
        Recommender.__init__(self, user_data_df, likes_df)

        opt = Adam(lr=1e-5, decay=1e-5 / 300)
        loss = "binary_crossentropy"

        m_f_model = load_model(self.M_F_MODEL_PATH, custom_objects={"keras":tf.keras})
        f_m_model = load_model(self.F_M_MODEL_PATH, custom_objects={"keras":tf.keras})

        m_f_model.compile(optimizer=opt, loss=loss)
        f_m_model.compile(optimizer=opt, loss=loss)

        self.m_f_model = m_f_model
        self.f_m_model = f_m_model
        self.image_shape = (self.IMAGE_DIM, self.IMAGE_DIM, 3)

    def calculate_reciprocal_preference(self, user_x_id, user_y_id):
        m_user_id, f_user_id = self.get_m_f_ids(user_x_id, user_y_id)
        m_f_scores = self.get_preference_scores(m_user_id, f_user_id, self.m_f_model)
        m_f_agg = self.aggregate_image_scores(m_f_scores)
        f_m_scores = self.get_preference_scores(f_user_id, m_user_id, self.f_m_model)
        f_m_agg = self.aggregate_image_scores(f_m_scores)

        reciprocal_score = self.aggregate_preference_scores(m_f_agg, f_m_agg)

        return reciprocal_score

    @staticmethod
    def aggregate_image_scores(scores):
        scores = np.nonzero(scores)
        return np.median(scores)

    def get_m_f_ids(self, user_x_id, user_y_id):
        if user_x_id in self.user_data_df.index:
            x_data = self.user_data_df.loc[user_x_id]
            if x_data['gender'] == "M":
                return user_x_id, user_y_id
            return user_y_id, user_x_id
        else:
            return user_x_id, user_y_id

    def get_preference_scores(self, user_x_id, user_y_id, model):
        liked_images, user_y_image = self.get_comparison_images(user_x_id, user_y_id)
        if len(liked_images) == 0:
            return [0.5]
        if len(liked_images) > 20:
            liked_images = liked_images[:20]

        images = [user_y_image] * len(liked_images)
        scores = model.predict((liked_images, images))
        print(scores)
        return scores

    def get_comparison_images(self, user_x_id, user_y_id):
        liked_users = self.get_liked_users(user_x_id)
        user_y_data = self.get_user_data(user_y_id)
        if len(liked_users) == 0 or len(user_y_data) == 0:
            return [], []

        image_ids = liked_users['image_id'].tolist()
        liked_images = map(self.read_image, image_ids)
        liked_images = self.remove_grayscale_images(liked_images)
        user_y_image = self.read_image(user_y_data['image_id'])

        if len(liked_images) == 0 or np.shape(user_y_image) != self.image_shape:
            return [], []

        return liked_images, user_y_image

    def remove_grayscale_images(self, images):
        colour_images = []

        for anchor in images:
            if np.shape(anchor) == self.image_shape:
                colour_images.append(anchor)

        return colour_images

    def read_image(self, image_id):
        # filename = "%i_%i" % (image_id, user_id)
        filename = str(image_id)
        path = "%s%s.jpg" % (self.IMAGES_FOLDER, str(filename))
        if not os.path.exists(path):
            return []

        img = imread(path, as_gray=False)
        img = resize(img, (self.IMAGE_DIM, self.IMAGE_DIM))

        return img
