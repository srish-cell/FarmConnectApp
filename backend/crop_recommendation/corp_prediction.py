import pickle


def recommend_crop(data):
    crop_recommendation_model_path = 'models/RandomForest.pkl'
    crop_recommendation_model = pickle.load(
        open(crop_recommendation_model_path, 'rb'))
    return crop_recommendation_model.predict(data)
data=[[85,58,41,29.42,78,7,227]]
print(recommend_crop(data))