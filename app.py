import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
import json
from dotenv import load_dotenv

app = FastAPI()


def load_models():
    model = CatBoostClassifier()
    model.load_model("catboost")
    return model


load_dotenv()
POSTGRES_ENGINE = os.environ.get("POSTGRES_ENGINE")


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(POSTGRES_ENGINE)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features(userid: int) -> pd.DataFrame:
    engine = create_engine(POSTGRES_ENGINE)
    user_data = pd.read_sql(f'SELECT * FROM public.user_data WHERE user_id={userid}', con=engine)

    data = pd.merge(post_info, user_data, how='cross')
    data = data.set_index(['user_id', 'post_id'])

    return data


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime = None, limit: int = 10) -> List[PostGet]:
    data = load_features(id)

    if not time:
        time = pd.Timestamp.today()

    data['weekend'] = time.weekday()
    data['hour'] = time.hour

    model = load_models()

    data['pred_proba'] = model.predict_proba(data)[:, 1]

    post_ids = list(data.sort_values('pred_proba', ascending=False).head(limit).reset_index().post_id)
    recs = post_texts.rename(columns={'post_id': 'id'}).set_index('id').loc[post_ids].reset_index()
    recs = recs.to_json(orient='records')
    recs = json.loads(recs)

    return recs


engine = create_engine(POSTGRES_ENGINE)

post_info = batch_load_sql('SELECT * FROM m_mishin_features_lesson_22_post')
post_texts = pd.read_sql('SELECT * FROM public.post_text_df', con=engine)
