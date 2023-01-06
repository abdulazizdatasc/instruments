import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
st.title("Qurollar, musiqa instrumentlari va tibbiy anjomlarni farqlovchi model")

file=st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    img=PILImage.create(file)
    
    model=load_learner('med_music_and_weapon.pkl')

    pred,pred_id,probs=model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)