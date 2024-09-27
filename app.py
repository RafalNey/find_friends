# Sekcja importowa
import json
import base64
import instructor
import pandas as pd
import streamlit as st
import plotly.express as px

from io import BytesIO
from openai import OpenAI
from pydantic import BaseModel
from dotenv import dotenv_values
from audiorecorder import audiorecorder
from pycaret.clustering import load_model, predict_model


# Tajny plik z kluczem (TYLKO lokalnie! Do testow przed wypuszczeniem w sina dal...)
env = dotenv_values('.env')

# Zmienne configurujace
AUDIO_TRANSCRIBE_MODEL = 'whisper-1'
DATA = './baza/welcome_survey_simple_v2.csv'
PICTURE = './pictures/meet-friends.png'
MODEL_NAME = './baza/welcome_survey_clustering_pipeline_v2'
CLUSTER_NAMES_AND_DESCRIPTIONS = './baza/welcome_survey_cluster_names_and_descriptions_v2.json'


# -------------------------------------------
#               BLOK FUNKCJI
# -------------------------------------------

# Sekcja funkcji wchodzacych do cache'u
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


@st.cache_data
def get_cluster_names_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, 'r') as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participant():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


# Inicjalizacja OpenAI clienta
def get_openai_client():
    return OpenAI(api_key=st.session_state['openai_api_key'])


# Sekcja funkcji transkrypcyjno-dzwiekowych
def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = 'audio.mp3'
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format='verbose_json',
    )

    return transcript.text


# Konwertuj tekst na audio
def text_to_audio(client, text):
    audio_file = BytesIO()
    response = client.audio.speech.create(
        model='tts-1',
        voice='onyx',
        input=text
    )
    audio_bytes = response.content
    audio_file = BytesIO(audio_bytes)
    audio_file.seek(0)
    return audio_file


# Funkcja do odtwarzania audio
def auto_play_audio(audio_file):
    audio_bytes = audio_file.getvalue()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# --------------------------------------------
#                     KOD GLOWNY
# --------------------------------------------


# Ladujemy OpenAI API key od uzytkownika
if not st.session_state.get('openai_api_key'):
    if 'OPENAI_API_KEY' in env:
        st.session_state['openai_api_key'] = env['OPENAI_API_KEY']
    else:
        st.info('Dodaj swoj klucz API OpenAI, by moc korzystac z tej aplikacji')
        st.session_state['openai_api_key'] = st.text_input('Klucz API', type='password')
        if st.session_state['openai_api_key']:
            st.rerun()

if not st.session_state.get('openai_api_key'):
    st.stop()


# Konfigurujemy z grubsza streamlita
st.set_page_config(
    page_title='Find Friends',
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------
#                     PASEK BOCZNY
# --------------------------------------------

with st.sidebar:
    # Tytul na pasku bocznym
    st.markdown('''
        <div style="text-align: center">
        <h1 style="font-size: 24px; margin: 0;">Meet friends in Data Scientists course based on the welcome survey</h1>
        </div>
        ''', unsafe_allow_html=True)

    st.header('Opowiedz nam coś o sobie')
    st.markdown('Pomozemy Ci znalezc osoby, które mają podobne do Twoich zainteresowania')

    age = st.selectbox('Wiek', ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    edu_level = st.selectbox('Wykształcenie', ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox('Ulubione zwierzęta', ['Brak ulubionych', 'Psy', 'Koty', 'Koty i psy', 'Inne'])
    fav_place = st.selectbox('Ulubione miejsce', ['Nad wodą', 'W górach', 'W lesie', 'Inne'])
    gender = st.radio('Płeć', ['Mężczyzna', 'Kobieta'])

    # Zbieranie danych do DataFrame
    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

    # Przewidywanie klastra na podstawie aktualnych danych
    model = get_model()
    cluster_data = predict_model(model, data=person_df)

    st.session_state['predicted_cluster_id'] = cluster_data['Cluster'].values[0]
    cluster_names_and_descriptions = get_cluster_names_descriptions()
    predicted_cluster_data = cluster_names_and_descriptions[st.session_state['predicted_cluster_id']]


# --------------------------------------------
#                     EKRAN GŁÓWNY
# --------------------------------------------

mowa_powitalna = '''
    Cześć! Jeśli szukasz przyjaciół na kursie Data Scientist, to dobrze trafiłeś.
    Moim zadaniem jest ci w tym pomóc. Najpierw opowiedz mi prosze trochę o sobie;
    ile masz lat, jakie masz wykształcenie, gdzie lubisz spędzać czas i czy masz
    jakieś swoje ulubione zwierzątko? Kiedy będziesz gotowy, żeby mi o tym
    wszystkim opowiedzieć, nagraj wiadomość.
    '''

client = get_openai_client()

# Odtwarzamy mowe powitalna, tylko jesli jeszcze nie zostala odtworzona
if 'welcome_message_played' not in st.session_state:
    st.session_state['welcome_message_played'] = False

if not st.session_state['welcome_message_played']:
    audio_file = text_to_audio(client, mowa_powitalna)
    auto_play_audio(audio_file)
    st.session_state['welcome_message_played'] = True

# Sprawdzanie nagrania uzytkownika
if 'note_audio_bytes' not in st.session_state:
    st.session_state['note_audio_bytes'] = None

if 'note_audio_text' not in st.session_state:
    st.session_state['note_audio_text'] = None

# Uzytkownik nagrywa swoja wiadomosc
st.title('Nagraj wiadomosc')
st.write('Podaj swoj wiek, plec, wyksztalcenie, ulubione zwierze i ulubione miejsce.')
note_audio = audiorecorder(
    start_prompt="Nagraj wiadomosc",
    stop_prompt="Zatrzymaj nagranie"
)

if note_audio:
    audio = BytesIO()
    note_audio.export(audio, format='mp3')
    st.session_state['note_audio_bytes'] = audio.getvalue()
    st.audio(st.session_state['note_audio_bytes'], format='audio/mp3')

    if st.session_state['note_audio_text']:
        st.text_area(
            'Jesli twoje informacje nie sa kompletne, nagraj sie jeszcze raz:',
            value=st.session_state['note_audio_text'],
            disabled=True,
        )

    if st.button('Zatwierdz wiadomosc'):
        st.session_state['note_audio_text'] = transcribe_audio(st.session_state['note_audio_bytes'])

# Jesli uzytkownik zatwierdzil wiadomosc, wyciagamy z niej wszystkie soki
if 'note_audio_text' in st.session_state and st.session_state['note_audio_text'] is not None:

    # Definicja modelu uzytkownika
    class PersonInfo(BaseModel):
        age: str
        edu_level: str
        fav_animals: str
        fav_place: str
        gender: str

    instructor_openai_client = instructor.from_openai(client)

    # Tresc promptu dla AI
    system_prompt = """
        Twoim zadaniem jest wyciągniecie informacji osobistych z podanego tekstu.
        Poniżej znajduje się przykładowa wypowiedź, z której należy wyciągnąć informacje.
        Wyciągnij wiek, płeć, poziom wykształcenia, ulubione miejsce i ulubione zwierzę.
        Wedlug nastepujacych kryteriow:
        wiek postaraj sie przypisac do 1 z nastepujacych kategorii:
        ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'], a nastepnie zapisz go do pola 'age'
        Wyksztalcenie postaraj sie przypisac do 1 z nastepujacych kategorii:
        ['Podstawowe', 'Średnie', 'Wyższe'], a nastepnie zapisz go do pola 'edu_level'
        Ulubione zwierze postaraj sie przypisac do 1 z nastepujacych kategorii:
        ['Brak ulubionych', 'Psy', 'Koty', 'Koty i psy', 'Inne'], a nastepnie zapisz go do pola 'fav_animals'
        Ulubione miejsce postaraj sie przypisac do 1 z nastepujacych kategorii:
        ['Nad wodą', 'W górach', 'W lesie', 'Inne'], a nastepnie zapisz go do pola 'fav_place'
        Płeć przypisz do 1 z nastepujacych kategorii:
        ['Mężczyzna', 'Kobieta'], a nastepnie zapisz go do pola 'gender'.

        Przykladowa wypowiedz:
        <user_text>
        Mam na imię Rafał, żyję już od poł wieku i czterech lat. Mam psa i lubię z nim spacerować po lesie.
        <user_text/>

        Przykladowe dane do wyciagniecia:
        <expected_data>
        {
        "age": "45-55"
        "edu_level": "NaN"
        "fav_place": "W lesie"
        "fav_animals": "Psy"
        "gender": "Mężczyzna"
        }
        <expected_data/>
    """

    audio_text = st.session_state['note_audio_text']
    user_input = audio_text

    user_info = instructor_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=PersonInfo,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""<user_text>{user_input}<user_text/>"""},
        ]
    )

    user_data = user_info.model_dump()

    # Zbieranie danych do DataFrame
    person_df = pd.DataFrame([{
        'age': user_data['age'],
        'edu_level': user_data['edu_level'],
        'fav_animals': user_data['fav_animals'],
        'fav_place': user_data['fav_place'],
        'gender': user_data['gender'],
    }])

    model = get_model()
    all_df = get_all_participant()
    cluster_names_and_descriptions = get_cluster_names_descriptions()

    predicted_cluster_id = predict_model(model, data=person_df)['Cluster'].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

    st.title(f'Najbliżej ci do grupy:    {predicted_cluster_data["name"]}')
    st.markdown(predicted_cluster_data['description'])

    mowa_pozegnalna = f'''
        Z tego, co widzę, najbliżej ci do grupy {predicted_cluster_data["name"]}.
        {predicted_cluster_data['description']}
        Jestem pewny, że wśród nich poznasz swoją bratnią duszę.
    '''

    st.image(PICTURE)
    audio_file = text_to_audio(client, mowa_pozegnalna)
    auto_play_audio(audio_file)

    same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
    st.metric('Liczba Twoich znajomych', len(same_cluster_df))

    st.header('Osoby z grupy')
    fig = px.histogram(same_cluster_df.sort_values('age'), x='age')
    fig.update_layout(
        title='Rozklad wieku w grupie',
        xaxis_title='Wiek',
        yaxis_title='Liczba osob',
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('edu_level'), x='edu_level')
    fig.update_layout(
        title='Rozklad wyksztalcenia w grupie',
        xaxis_title='Wyksztalcenie',
        yaxis_title='Liczba osob',
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('fav_animals'), x='fav_animals')
    fig.update_layout(
        title='Rozklad ulubionych zwierzat w grupie',
        xaxis_title='Ulubione zwierzeta',
        yaxis_title='Liczba osob',
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('fav_place'), x='fav_place')
    fig.update_layout(
        title='Rozklad ulubionych miejsc w grupie',
        xaxis_title='Ulubione miejsce',
        yaxis_title='Liczba osob',
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('gender'), x='gender')
    fig.update_layout(
        title='Rozklad plci w grupie',
        xaxis_title='Plec',
        yaxis_title='Liczba osob',
    )
    st.plotly_chart(fig)
