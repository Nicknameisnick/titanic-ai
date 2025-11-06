#!/usr/bin/env python
# coding: utf-8

# In[37]:


# Importeer nodige libraries
import pandas as pd
import requests
import plotly.express as px 
import folium
import numpy as np 
import streamlit as st 
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from folium.plugins import MarkerCluster


# In[38]:


# Verkrijg data uit de API voor laadpaaldata
response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=59843&compact=true&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")
responsejson  = response.json()

laadpalen_data = pd.json_normalize(responsejson)
data_laadpaaltijd4 = pd.json_normalize(laadpalen_data.Connections)
data_laadpaaltijd5 = pd.json_normalize(data_laadpaaltijd4[0])


# In[39]:


# Zet de data in een set
Laadpalen = pd.concat([laadpalen_data, data_laadpaaltijd5], axis=1)
Laadpalen.head()
len(Laadpalen)


# In[40]:


Laadpalen.info()


# In[41]:


Laadpalen.describe()


# In[42]:


Laadpalen.columns


# In[43]:


# Verwijder de kolommen met contactinformatie
Laadpalen = Laadpalen.drop(['AddressInfo.ContactTelephone1', 'AddressInfo.ContactTelephone2', 
                            'AddressInfo.ContactEmail', 'AddressInfo.RelatedURL'], axis=1)


# In[44]:


def plot_missing_values():
    # Bereken het percentage ontbrekende waarden per kolom
    missing_values_percentage = Laadpalen.isna().sum() / len(Laadpalen) * 100

    # Zet de data in een DataFrame voor gebruik in Plotly
    missing_data_laadpaaltijd = pd.DataFrame({
        'Column': missing_values_percentage.index,
        'Percentage': missing_values_percentage.values
    })

    # Maak de barplot met Plotly
    fig = px.bar(missing_data_laadpaaltijd, x='Column', y='Percentage', title='Percentage of Missing Values per Column',
                 labels={'Percentage': 'Percentage (%)', 'Column': 'Columns'},
                 color='Percentage',
                 color_continuous_scale='YlOrRd')

    # Pas de layout aan
    fig.update_layout(
        xaxis_title="Columns",
        yaxis_title="Percentage (%)",
        title_x=0.5,  # Centreer de titel
        template="plotly_white"
    )

    return fig
plot_missing_values()


# In[45]:


# Drop NAN values zodat je alleen de comments kan zien
Laadpalen['AddressInfo.AccessComments'].dropna()


# In[46]:


# Drop NAN values zodat je alleen de comments kan zien
Laadpalen['GeneralComments'].dropna()


# In[47]:


missing_values_percentage = Laadpalen.isna().sum() / len(Laadpalen) * 100

# Identify columns to drop (those with more than 30% missing values)
columns_to_drop = missing_values_percentage[missing_values_percentage > 30].index

# Drop the identified columns from the DataFrame
laadpalen_filtered1 = Laadpalen.drop(columns=columns_to_drop)
laadpalen_filtered1.head()


# In[48]:


laadpalen_filtered1.isna().sum()


# In[49]:


laadpalen_filtered1 = laadpalen_filtered1.drop(columns=[
    'DataProvidersReference', 
    'OperatorID', 
    'OperatorsReference',
    'LevelID', 
    'ConnectionTypeID',
])

laadpalen_filtered1.columns


# In[50]:


laadpalen_filtered1.isna().sum()


# In[51]:


# Change NaN-values in 'Voltage' and 'PowerKW with median or mean
laadpalen_filtered1['PowerKW'].fillna(laadpalen_filtered1['PowerKW'].median(), inplace=True)
laadpalen_filtered1.head()


# In[52]:


# Drop the rest of the missing values since they cant be filled
laadpalen_filtered2 = laadpalen_filtered1.dropna()
len(laadpalen_filtered2)


# In[53]:


import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load the GeoPackage (make sure to provide the correct path and layer name)
# Example layer name 'provinces' can be changed according to your actual layer name
provinces_gpkg = gpd.read_file('gadm41_NLD.gpkg', layer='ADM_ADM_1')

# Convert the DataFrame into a GeoDataFrame with geometry (points based on lat/lon)
geometry = [Point(xy) for xy in zip(laadpalen_filtered2['AddressInfo.Longitude'], laadpalen_filtered2['AddressInfo.Latitude'])]
geo_df = gpd.GeoDataFrame(laadpalen_filtered2, geometry=geometry)

# Set the Coordinate Reference System (CRS) of the points to WGS84 (EPSG:4326)
geo_df.set_crs(epsg=4326, inplace=True)

# Ensure the provinces GeoPackage is also in the same CRS, or convert it
if provinces_gpkg.crs != geo_df.crs:
    provinces_gpkg = provinces_gpkg.to_crs(geo_df.crs)

# Perform a spatial join between your points and the province polygons
# 'predicate' ensures the points fall within the province boundaries
joined = gpd.sjoin(geo_df, provinces_gpkg, how='left', predicate='intersects')

# Assign the province to the original DataFrame based on the spatial join
# Replace 'province_column_name' with the actual name of the province column in the GeoPackage
laadpalen_filtered2['Province'] = joined['NAME_1']

# Display the updated DataFrame
laadpalen_filtered2.head()



# In[54]:


laadpalen_filtered2['Province'].unique()


# In[55]:


# Define a mapping for standardizing province names
province_mapping = {
    'Fryslân': 'Friesland'              # Standardizing Fryslân to Friesland
}

# Replace the province names in the DataFrame
laadpalen_filtered2['Province'] = laadpalen_filtered2['Province'].replace(province_mapping)


# In[56]:


# Define a color mapping for each province
province_colors = {
    'Drenthe': 'blue',
    'Flevoland': 'green',
    'Friesland': 'purple',
    'Gelderland': 'orange',
    'Groningen': 'red',
    'Limburg': 'darkblue',
    'North-Brabant': 'darkgreen',
    'Noord-Holland': 'darkred',
    'Overijssel': 'pink',
    'Utrecht': 'lightblue',
    'Zeeland': 'lightgreen',
    'Zuid-Holland': 'yellow',
}


# In[57]:


# Define province coordinates for centering and zoom level
province_coords = {
    'Drenthe': [52.8, 6.6],
    'Flevoland': [52.5, 5.6],
    'Friesland': [53.1, 5.8],
    'Gelderland': [52.0, 5.9],
    'Groningen': [53.2, 6.6],
    'Limburg': [51.4, 6.0],
    'Noord-Brabant': [51.6, 5.2],
    'Noord-Holland': [52.7, 4.8],
    'Overijssel': [52.5, 6.3],
    'Utrecht': [52.1, 5.2],
    'Zeeland': [51.5, 3.9],
    'Zuid-Holland': [52.0, 4.5],
}


# In[58]:


def map_plot(province):
    # Get coordinates and set zoom level based on the selected province
    if province in province_coords:
        location = province_coords[province]
        zoom_start = 10  # Zoom in more for specific province
    else:
        location = [52.1326, 5.2913]  # Default center for Netherlands
        zoom_start = 7  # General zoom for all

    # Create a base map centered at the selected province
    m = folium.Map(location=location, zoom_start=zoom_start)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Filter the data based on selected province
    if province != 'All':
        filtered_data = laadpalen_filtered2[laadpalen_filtered2['Province'] == province]
    else:
        filtered_data = laadpalen_filtered2

    # Add markers to the cluster with unique colors
    for _, row in filtered_data.iterrows():
        popup_content = f"""
        <b>Address:</b> {row['AddressInfo.Title']}<br>
        <b>Town:</b> {row['AddressInfo.Town']}<br>
        <b>Is Recently Verified:</b> {row['IsRecentlyVerified']}<br>
        <b>Number of Points:</b> {row['NumberOfPoints']}
        """
        
        province_name = row['Province']
        color = province_colors.get(province_name, 'gray')  # Default to gray if not found
        
        # Add marker to the cluster
        folium.Marker(
            location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    
    return m


# In[59]:


# Lees laadpaaldata dataset in
data_laadpaaltijd = pd.read_csv('laadpaaldata.csv', sep=',')
data_laadpaaltijd.head()


# In[60]:


data_laadpaaltijd.isna().sum()


# In[61]:


data_laadpaaltijd['ConnectedTime_min'] = data_laadpaaltijd['ConnectedTime'] * 60
data_laadpaaltijd.head()


# In[62]:


def histogram_contime(data_laadpaaltijd):
    # Checkbox to remove outliers
    remove_outliers = st.checkbox("Remove outliers")
    
    if remove_outliers:
        # Calculate the IQR for outlier detection
        Q1 = data_laadpaaltijd['ConnectedTime'].quantile(0.25)
        Q3 = data_laadpaaltijd['ConnectedTime'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data to remove outliers
        data_laadpaaltijd = data_laadpaaltijd[(data_laadpaaltijd['ConnectedTime'] >= lower_bound) & 
                                               (data_laadpaaltijd['ConnectedTime'] <= upper_bound)]

    # Plotting the histogram of ConnectedTime
    fig = px.histogram(data_laadpaaltijd, 
                       x="ConnectedTime", 
                       nbins=50,  # Adjusted number of bins for better readability
                       title="Histogram of Connected Time in Hours", 
                       range_x=[0, max(data_laadpaaltijd['ConnectedTime'])],
                       labels={'ConnectedTime': 'Connected Time (Hours)'})  # Axis label

    # Adding gridlines for better readability
    fig.update_xaxes(title_text='Connected Time (Hours)', title_font=dict(size=14))
    fig.update_yaxes(title_text='Frequency', title_font=dict(size=14))
    fig.update_layout(title_font=dict(size=16),  # Title font size
                      xaxis_tickfont=dict(size=12),  # X-axis tick font size
                      yaxis_tickfont=dict(size=12))  # Y-axis tick font size

    # Customize the color and add black outlines to each bin
    fig.update_traces(marker=dict(color='royalblue', 
                                   opacity=0.7,
                                   line=dict(color='black', width=1)))  # Change the color and opacity with black outlines

    # Display the histogram in Streamlit
    st.plotly_chart(fig)


# In[63]:


# Your plotting function
def energy_plot(data_laadpaaltijd):
    # Omzetten naar datetime en NaT opvangen
    data_laadpaaltijd['Started'] = pd.to_datetime(data_laadpaaltijd['Started'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Maand en week toevoegen aan de dataframe
    data_laadpaaltijd['Month'] = data_laadpaaltijd['Started'].dt.month_name()
    data_laadpaaltijd['Week'] = data_laadpaaltijd['Started'].dt.isocalendar().week

    # Filter rijen zonder geldige datums (NaT)
    data_laadpaaltijd = data_laadpaaltijd.dropna(subset=['Started'])

    # TotalEnergy groeperen op maand en week
    monthly_total_energy = data_laadpaaltijd.groupby(['Month'], as_index=False)['TotalEnergy'].sum()
    weekly_total_energy = data_laadpaaltijd.groupby(['Week'], as_index=False)['TotalEnergy'].sum()

    # Idle en verloren energie berekenen
    idle_power = 0.05  # Standby vermogen in kW
    data_laadpaaltijd['ConnectedTime'] = data_laadpaaltijd['ConnectedTime'].astype(float)
    data_laadpaaltijd['ChargeTime'] = data_laadpaaltijd['ChargeTime'].astype(float)
    data_laadpaaltijd['IdleTime'] = data_laadpaaltijd['ConnectedTime'] - data_laadpaaltijd['ChargeTime']  # Idle time in uren
    data_laadpaaltijd['LostEnergy'] = data_laadpaaltijd['IdleTime'] * idle_power  # Verloren energie in kWh

    # LostEnergy groeperen op maand en week
    monthly_lost_energy = data_laadpaaltijd.groupby(['Month'], as_index=False)['LostEnergy'].sum()
    weekly_lost_energy = data_laadpaaltijd.groupby(['Week'], as_index=False)['LostEnergy'].sum()

    # Zorg ervoor dat de maanden in de juiste volgorde zijn
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_total_energy['Month'] = pd.Categorical(monthly_total_energy['Month'], categories=months_order, ordered=True)
    monthly_lost_energy['Month'] = pd.Categorical(monthly_lost_energy['Month'], categories=months_order, ordered=True)

    # Sorteren op maand
    monthly_total_energy = monthly_total_energy.sort_values('Month')
    monthly_lost_energy = monthly_lost_energy.sort_values('Month')

    # Maak de basisgrafiek met de verschillende traces voor TotalEnergy en LostEnergy (maandelijks en wekelijks)
    fig = go.Figure()

    # Voeg de Total Energy traces toe
    fig.add_trace(go.Scatter(x=monthly_total_energy['Month'], y=monthly_total_energy['TotalEnergy'], mode='lines+markers', name='Monthly TotalEnergy', visible=True))
    fig.add_trace(go.Scatter(x=weekly_total_energy['Week'], y=weekly_total_energy['TotalEnergy'], mode='lines+markers', name='Weekly TotalEnergy', visible=False))

    # Voeg de Lost Energy traces toe in het rood
    fig.add_trace(go.Scatter(x=monthly_lost_energy['Month'], y=monthly_lost_energy['LostEnergy'], mode='lines+markers', name='Monthly LostEnergy', line=dict(color='red'), visible=False))
    fig.add_trace(go.Scatter(x=weekly_lost_energy['Week'], y=weekly_lost_energy['LostEnergy'], mode='lines+markers', name='Weekly LostEnergy', line=dict(color='red'), visible=False))

    # Update layout met 1 dropdown voor zowel Total als Lost Energy
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label="Total Energy - Monthly", method="update", 
                     args=[{"visible": [True, False, False, False]}, 
                           {"xaxis": {"title": "Month of the Year"}}]),
                dict(label="Total Energy - Weekly", method="update", 
                     args=[{"visible": [False, True, False, False]}, 
                           {"xaxis": {"title": "Week Number"}}]),
                dict(label="Lost Energy - Monthly", method="update", 
                     args=[{"visible": [False, False, True, False]}, 
                           {"xaxis": {"title": "Month of the Year"}}]),
                dict(label="Lost Energy - Weekly", method="update", 
                     args=[{"visible": [False, False, False, True]}, 
                           {"xaxis": {"title": "Week Number"}}])
            ],
            direction="down",
            showactive=True,
            x=0.5,  # Geplaatst in het midden
            xanchor="center",
            y=1.15,
            yanchor="top"
        )]
    )

    # Algemene layout van de grafiek
    fig.update_layout(
        title="Energy Statistics (Total vs Lost)",
        xaxis_title="Time Period",
        yaxis_title="Energy (kWh)",
        template='plotly_white'
    )

    # Grafiek weergeven
    return fig


# In[64]:


def start_end_time():
    # Convert columns to datetime with day first
    data_laadpaaltijd['Started'] = pd.to_datetime(data_laadpaaltijd['Started'], dayfirst=True, errors='coerce')
    data_laadpaaltijd['Ended'] = pd.to_datetime(data_laadpaaltijd['Ended'], dayfirst=True, errors='coerce')

    # New columns for just the hours (24-hour format)
    data_laadpaaltijd['start_time'] = data_laadpaaltijd['Started'].dt.hour
    data_laadpaaltijd['end_time'] = data_laadpaaltijd['Ended'].dt.hour

    # Drop rows with NaT in Started or Ended
    data_laadpaaltijd2 = data_laadpaaltijd.dropna(subset=['Started', 'Ended'])

    # Create histograms for start_time and end_time
    start_hist = px.histogram(data_laadpaaltijd2, x='start_time', nbins=24).data[0]
    end_hist = px.histogram(data_laadpaaltijd2, x='end_time', nbins=24).data[0]

    # Create the figure
    fig = go.Figure()

    # Add start_time histogram as the default trace
    fig.add_trace(start_hist)

    # Add dropdown layout
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(
                    args=[{'x': [start_hist.x], 'y': [start_hist.y], 'name': 'Start Time'}],
                    label="Start Time",
                    method="update"
                ),
                dict(
                    args=[{'x': [end_hist.x], 'y': [end_hist.y], 'name': 'End Time'}],
                    label="End Time",
                    method="update"
                )
            ],
            direction="down",
            showactive=True,
        )],
        xaxis_title='Hour of the Day',
        yaxis_title='Count',
        title="Histogram of Start and End Times",
        bargap=0.1  # Space between bars
    )

    # Adjust x-axis ticks to show all hours of the day
    fig.update_xaxes(tickvals=list(range(24)))

    # Display the plot in Streamlit
    st.plotly_chart(fig)



# In[65]:


data_cars = pd.read_pickle('cars.pkl')
data_cars.head()


# In[66]:


data_cars.isna().sum()


# In[67]:


# Functie om brandstofsoort te bepalen
def bepaal_brandstof(naam):
    naam = naam.lower()
    if any(keyword in naam for keyword in ['edrive', 'id', 'ev', 'electric', 'atto', 'pro', 'ex', 'model', 'e-tron', 'mach-e', 'kw']):
        return 'elektrisch'
    elif any(keyword in naam for keyword in ['hybrid', 'phev', 'plugin']):
        return 'hybride'
    elif 'diesel' in naam:
        return 'diesel'
    elif 'waterstof' in naam or 'fuel cell' in naam:
        return 'waterstof'
    else:
        return 'benzine'  # Default voor auto's die geen andere trefwoorden bevatten

# Pas de functie toe om de brandstofsoort te bepalen
data_cars['brandstof'] = data_cars['handelsbenaming'].apply(bepaal_brandstof)

# Bekijk de eerste paar rijen met de nieuwe 'brandstof' kolom
data_cars[['handelsbenaming', 'brandstof']].head()


# In[68]:


def auto_per_maand():
    # Ensure the 'datum_eerste_toelating' column is in datetime format
    data_cars['datum_eerste_toelating'] = pd.to_datetime(data_cars['datum_eerste_toelating'], format='%Y%m%d', errors='coerce')

    # Group the data by 'datum_eerste_toelating' and 'brandstof', and count the number of cars per group
    grouped_data = data_cars.groupby(['datum_eerste_toelating', 'brandstof']).size().unstack(fill_value=0)

    # Reset the index to use the data for Plotly
    grouped_data = grouped_data.reset_index()

    # Calculate the cumulative sum for each fuel type
    grouped_data[['elektrisch', 'benzine']] = grouped_data[['elektrisch', 'benzine']].cumsum()

    # Melt the data to long-form format suitable for Plotly
    melted_data = grouped_data.melt(id_vars='datum_eerste_toelating', var_name='brandstof', value_name='aantal_autos')

    # Add a slider to select the month
    min_date = melted_data['datum_eerste_toelating'].min().to_pydatetime()  # Convert to datetime
    max_date = melted_data['datum_eerste_toelating'].max().to_pydatetime()  # Convert to datetime

    # Slider for month selection
    selected_date = st.slider(
        "Selecteer een maand",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )

    # Filter the data based on the selected month
    filtered_data = melted_data[(melted_data['datum_eerste_toelating'] >= selected_date[0]) & 
                                (melted_data['datum_eerste_toelating'] <= selected_date[1])]

    # Create a line plot with separate lines for 'elektrisch' and 'benzine'
    fig_line = px.line(filtered_data, 
                        x='datum_eerste_toelating', 
                        y='aantal_autos', 
                        color='brandstof',
                        color_discrete_map={'benzine': 'blue', 'elektrisch': 'red'},
                        labels={'aantal_autos': 'Aantal auto\'s', 'datum_eerste_toelating': 'Datum eerste toelating'}, 
                        title=f'Cumulatief aantal auto\'s per brandstofsoort van {selected_date[0].strftime("%Y-%m")} tot {selected_date[1].strftime("%Y-%m")}'
    )

    # Create a histogram of fuel types
    color_map = {
        'benzine': 'blue',
        'elektrisch': 'red'
    }

    # Update filtered_data for histogram to only include relevant fuel types
    filtered_hist_data = data_cars[(data_cars['brandstof'].isin(color_map.keys())) & 
                                    (data_cars['datum_eerste_toelating'] >= selected_date[0]) & 
                                    (data_cars['datum_eerste_toelating'] <= selected_date[1])]

    # Create the histogram
    fig_hist = px.histogram(filtered_hist_data, 
                             x="brandstof", 
                             title="Histogram of Fueltypes in Cars", 
                             color='brandstof',  # Set color based on fuel type
                             color_discrete_map=color_map,
                             category_orders={'brandstof': list(color_map.keys())},  # Ensure specific order
                             text_auto=True)  # Optional: to display counts on bars

    # Update layout for better appearance
    fig_hist.update_layout(bargap=0.2)  # Optional: Adjust gap between bars

    # You can set a specific range or derive it from the original data
    max_count = data_cars['brandstof'].value_counts().max()  # Get the max count for y-axis scaling
    fig_hist.update_yaxes(range=[0, max_count])  # Set y-axis limits

    # Create two columns to display plots side by side
    col1, col2 = st.columns(2)

    # Display the line plot in the first column
    with col1:
        st.plotly_chart(fig_line)  # Show the Plotly line chart in Streamlit

    # Display the histogram in the second column
    with col2:
        st.plotly_chart(fig_hist)  # Show the histogram in Streamlit


# In[69]:


# Select relevant columns for analysis
data_cars2 = data_cars[['massa_ledig_voertuig', 
                         'datum_eerste_toelating', 
                         'aantal_deuren', 
                         'lengte',
                         'catalogusprijs', 
                         'breedte', 
                         'hoogte_voertuig',  
                         'wielbasis']]


# In[70]:


def corr_plot():
    # Calculate the correlation matrix and round the values
    corr_matrix = data_cars2.corr()
    corr_values = np.round(corr_matrix.values, 2)  # Round the correlation values to 2 decimal places

    # Create the heatmap using Plotly
    heatmap = go.Heatmap(
        z=corr_values,  # Correlation matrix values
        x=corr_matrix.columns,  # Columns for x-axis labels
        y=corr_matrix.index,  # Index for y-axis labels
        colorscale='thermal',  # Set the color scale
        colorbar=dict(title="Correlation")  # Add a colorbar title
    )

    # Create annotations to display the correlation values on the heatmap
    annotations = []
    for i in range(corr_values.shape[0]):
        for j in range(corr_values.shape[1]):
            annotations.append(
                dict(
                    x=corr_matrix.columns[j],  # x position of the cell
                    y=corr_matrix.index[i],  # y position of the cell
                    text=str(corr_values[i, j]),  # Display the correlation value as text
                    showarrow=False,  # Disable arrows for annotations
                    font=dict(color="black")  # Set the text color to black
                )
            )

    # Create the layout, including the annotations for the heatmap
    layout = go.Layout(
        title='Correlation Matrix for Categorical and Numerical Variables',
        annotations=annotations  # Add the annotations to the layout
    )

    # Create the figure
    fig = go.Figure(data=[heatmap], layout=layout)
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)



# In[71]:


def train_model():
    # Drop missing values
    train = data_cars2.dropna()

    # Define the target variable 'catalogusprijs'
    if 'catalogusprijs' not in train.columns:
        st.error("Error: 'catalogusprijs' is not in train DataFrame.")
        return
    
    y = train["catalogusprijs"]

    # Select features for the model
    features = ["massa_ledig_voertuig", "lengte", "wielbasis", "breedte"]
    X = train[features]  # Training data

    # Initialize and train the RandomForest model
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=1)
    model.fit(X, y)

    # Make predictions for the training set
    train_predictions = model.predict(X)

    # Create a density plot using Plotly
    fig = go.Figure()

    # Density of Actual Catalogusprijs
    fig.add_trace(go.Histogram(
        x=y,
        histnorm='probability density',
        name='Actual Catalogusprijs',
        opacity=0.5,
        marker=dict(color='blue'),
        nbinsx=50
    ))

    # Density of Predicted Catalogusprijs
    fig.add_trace(go.Histogram(
        x=train_predictions,
        histnorm='probability density',
        name='Predicted Catalogusprijs',
        opacity=0.5,
        marker=dict(color='orange'),
        nbinsx=50
    ))

    # Update layout
    fig.update_layout(
        title='Density Plot of Actual vs Predicted Catalogusprijs',
        xaxis_title='Catalogusprijs',
        yaxis_title='Density',
        barmode='overlay',
        showlegend=True
    )

    # Display the figure in the Streamlit app
    st.plotly_chart(fig)

