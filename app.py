import streamlit as st
import pandas as pd
import joblib
import folium
import plotly.express as px
from streamlit_folium import folium_static


# Load data and model
try:
    model = joblib.load('https://drive.google.com/file/d/1IZ1zMzy7-WOvA1JNPfMQWPl1Ch7HtToV/view?usp=sharing')
    encoders = joblib.load("https://drive.google.com/file/d/16d3iLlkwK7dpA5AOBATTukntnUymfpdv/view?usp=sharing")
    df = pd.read_csv("https://drive.google.com/file/d/1jOcjb5T3Lmu2UdKW04XULK64wpAk1Hzo/view?usp=sharing")
except FileNotFoundError:
    st.error("Error: One or more files not found. Please check file paths.")
    # Set default values if data loading fails
    model = None
    encoders = {}
    df = pd.DataFrame()

def get_distance(origin, dest):
    distances = df[(df['origin'] == origin) & (df['dest'] == dest)]['distance'].values
    return distances[0] if distances.any() else 0

# Custom slider component
def custom_slider(label, min_value, max_value, default_value, key):
    st.markdown(label)
    value = st.slider("", min_value, max_value, default_value, key=key)
    return value

def plot_flight_routes(origin, dest):
    # Geospatial Information: Flight routes map
    flight_map = folium.Map(location=[37.7749, -122.4194], zoom_start=4)  # Default to the center of the USA

    # Add markers for origin and destination airports
    folium.Marker(location=[37.7749, -122.4194], popup='San Francisco (SFO)').add_to(flight_map)
    folium.Marker(location=[40.6413, -73.7781], popup='New York (JFK)').add_to(flight_map)

    # Draw a line between origin and destination
    folium.PolyLine(locations=[[37.7749, -122.4194], [40.6413, -73.7781]], color="blue", weight=2.5, opacity=1).add_to(flight_map)

    return flight_map

def app():
    st.title('Flight Delay Prediction')

    # Display the first few rows of the DataFrame
    st.write(df.head())

    # User input elements
    carrier = st.selectbox('Carrier', sorted([''] + list(df['carrier'].unique())))
    origin = st.selectbox('Origin', sorted([''] + list(df['origin'].unique())))
    dest = st.selectbox('Destination', sorted([''] + list(df['dest'].unique())))

    # Automatically fill distance based on selected origin and destination
    distance = get_distance(origin, dest)

    # Display the distance as text
    st.text(f'Distance (miles): {distance}')

    # Display map after choosing origin and destination
    if origin and dest and distance != 0:
        st.subheader('Flight Routes Map')
        flight_map = plot_flight_routes(origin, dest)
        folium_static(flight_map)

    # User input sliders
    hour_changed = custom_slider('Scheduled Departure Hour (0-23)', 0, 23, 0, key="hour_slider")
    day_changed = custom_slider('Scheduled Departure Day (1-31)', 1, 31, 1, key="day_slider")
    month_changed = custom_slider('Scheduled Departure Month (1-12)', 1, 12, 1, key="month_slider")

    

    # Initialize session state
    if 'sliders_changed' not in st.session_state:
        st.session_state.sliders_changed = False

    # Update session state based on slider changes
    if hour_changed != 0 or day_changed != 1 or month_changed != 1:
        st.session_state.sliders_changed = True

    # Button to trigger prediction
    if st.button('Predict'):
        try:
            if carrier == '' or origin == '' or dest == '':
                st.warning('Please fill in all the required fields (Carrier, Origin, Destination).')
            elif distance == 0:
                st.warning('This flight does not travel between the selected origin and destination.')
            elif not st.session_state.sliders_changed:
                st.warning('Please move at least one slider for hour, day, or month.')
            else:
                input_data = pd.DataFrame({'carrier': [carrier], 'origin': [origin], 'dest': [dest],
                                           'distance': [distance], 'hour': [hour_changed], 'day': [day_changed], 'month': [month_changed]})
                for col in encoders.keys():
                    input_data[col] = encoders[col].transform(input_data[col])[0]

                if model is not None:
                    prediction = model.predict(input_data.values)
                    if prediction[0] == 1:
                        st.write('This flight is likely to be departing late. Thank you for your Cooperation.')
                    else:
                        st.write('This flight is likely to be departing on time')

                    # Visualization: Distribution of delays for the selected carrier
                    delays = df[df['carrier'] == carrier]['dep_delay'].dropna()
                    fig_delay_distribution = px.histogram(delays, x='dep_delay', nbins=50, title=f'Distribution of Delays for {carrier}')
                    st.plotly_chart(fig_delay_distribution.update_layout())

                    # Visualization: Scatter plot of departure hour vs. delays
                    fig_scatter = px.scatter(df, x='hour', y='dep_delay', title='Departure Hour vs. Delays')
                    st.plotly_chart(fig_scatter.update_layout())

                   # Geospatial Information: Flight routes map
                    flight_map = plot_flight_routes(origin, dest)
                    folium_static(flight_map)

                else:
                    st.error('Model not loaded successfully. Please check your setup.')
        except Exception as e:
            st.error(f'An error occurred during prediction: {str(e)}')

if __name__ == '__main__':
    app()
