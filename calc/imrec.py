from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .recommender import Recommender

from skimage.io import imread
from skimage.transform import resize

import os

class ImRec(Recommender):

    IMAGES_FOLDER = "/Users/james.neve/Data/Preference/TestUserImages/"
    IMAGE_DIM = 100
    M_F_MODEL_PATH = "/Users/james.neve/development/python/recon_with_images/models/face_data_siamese_m_200330.h5"

    def __init__(self, user_data_df, likes_df):
        Recommender.__init__(self, user_data_df, likes_df)

        opt = Adam(lr=1e-5, decay=1e-5 / 300)
        loss = "binary_crossentropy"

        m_f_model = load_model(self.M_F_MODEL_PATH, custom_objects={"keras":tf.keras})
        # f_m_model = load_model("../models/face_data_siamese_f_200330.h5")

        m_f_model.compile(optimizer=opt, loss=loss)
        # f_m_model.compile(optimizer=opt, loss=loss)

        self.m_f_model = m_f_model
        # self.f_m_model = f_m_model

    def calculate_reciprocal_preference(self, m_user_id, f_user_id):
        m_f_score = self.get_preference_score(m_user_id, f_user_id, self.m_f_model)
        # f_m_score = self.get_preference_score(f_user_id, m_user_id, self.f_m_model)
        f_m_score = 0.5

        reciprocal_score = self.aggregate_preference_scores(m_f_score, f_m_score)

        return reciprocal_score

    def get_preference_score(self, user_x_id, user_y_id, model):
        images = self.get_comparison_images(user_x_id, user_y_id)
        if len(images) == 0:
            return 0.5

        score = model.predict([images[0]], [images[1]])
        return score


    def get_comparison_images(self, user_x_id, user_y_id):
        liked_users = self.get_liked_users(user_x_id)
        user_y_data = self.get_user_data(user_y_id)
        if len(liked_users) == 0 or len(user_y_data) == 0:
            return []

        image_id = liked_users['image_id'].tolist()[0]
        anchor = self.read_image(image_id)
        user_y_image = self.read_image(user_y_data['image_id'])
        print(anchor)
        print(user_y_image)

        if len(anchor) == 0 or len(user_y_image) == 0:
            return []

        return anchor, user_y_image

    def read_image(self, image_id):
        path = "%s%s.jpg" % (self.IMAGES_FOLDER, str(image_id))
        if not os.path.exists(path):
            return []

        img = imread(path, as_gray=False)
        img = resize(img, (self.IMAGE_DIM, self.IMAGE_DIM))

        return img
