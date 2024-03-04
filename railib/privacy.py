from math import radians
import calendar
import itertools as it
import pkg_resources


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import folium
from folium import plugins
import geopy
from geopy import Nominatim
from geopy.exc import GeocoderTimedOut
import datetime as dt

TILES = "Cartodb Positron"

NYC_CENTER = np.array([40.7128, -74.0060])

geolocator = Nominatim()
geopy.geocoders.options.default_user_agent = "ai-law"


def distance_poi(rides_df, poi_coords, mode="pickup"):

    earth_mean_radius = 6371000  # meters

    assert mode in ("pickup", "dropoff")

    rides_radians = (
        rides_df[[f"{mode}_lat", f"{mode}_lng"]].applymap(radians).to_numpy()
    )
    poi_radians = np.array([radians(poi_coords[0]), radians(poi_coords[1])])[None, :]

    return (haversine_distances(rides_radians, poi_radians) * earth_mean_radius)[:, 0]


def read_taxi_data(path, source):
    assert source in ("yellow", "green", "old_yellow")

    if source == "yellow":
        cols = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
    elif source == "green":
        cols = [
            "lpep_pickup_datetime",
            "Lpep_dropoff_datetime",
            "Passenger_count",
            "Pickup_longitude",
            "Pickup_latitude",
            "Dropoff_longitude",
            "Dropoff_latitude",
        ]
    elif source == "old_yellow":
        cols = [
            " pickup_datetime",
            " dropoff_datetime",
            " passenger_count",
            " pickup_longitude",
            " pickup_latitude",
            " dropoff_longitude",
            " dropoff_latitude",
        ]

    return (
        pd.read_csv(path, index_col=False, parse_dates=cols[:2], usecols=cols)[cols]
        .assign(type_="yellow")
        .rename(
            dict(
                zip(
                    cols,
                    [
                        "pickup_dt",
                        "dropoff_dt",
                        "n_passangers",
                        "pickup_lng",
                        "pickup_lat",
                        "dropoff_lng",
                        "dropoff_lat",
                    ],
                )
            ),
            axis=1,
        )
        .dropna()
    )


def plot_hourly(dt_series):
    ax = (dt_series.dt.hour.value_counts(normalize=True).sort_index() * 100).plot(
        xticks=range(24), figsize=(12, 6), grid=True
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("% Daily Rides")
    ax.set_title("Precentage of NYC Taxi (Yellow & Green) Daily Rides per Hour of Day")

    return ax


def plot_heatmap(df_lat_lng):
    pickup_hm = folium.Map(location=NYC_CENTER, tiles=TILES, zoom_start=12)

    (plugins.HeatMap(df_lat_lng.to_numpy(), radius=5, blur=5).add_to(pickup_hm))

    return pickup_hm


def find_rides_between_two_coords(rides_df, pickup_coords, dropoff_coords, radius=500):
    pickup_mask = distance_poi(rides_df, pickup_coords, "pickup") <= radius
    dropoff_mask = distance_poi(rides_df, dropoff_coords, "dropoff") <= radius
    rides_mask = pickup_mask & dropoff_mask

    return rides_df[rides_mask]


def build_duration_table_viz(rides_df, pickup_address, dropoff_address, radius=500):
    pickup_location = geolocator.geocode(pickup_address)
    dropoff_location = geolocator.geocode(dropoff_address)

    pickup_coords = (pickup_location.latitude, pickup_location.longitude)
    dropoff_coords = (dropoff_location.latitude, dropoff_location.longitude)

    rides_between = find_rides_between_two_coords(
        rides_df, pickup_coords, dropoff_coords, radius
    ).copy()

    rides_between["Duration"] = (
        rides_between["dropoff_dt"] - rides_between["pickup_dt"]
    ).dt.seconds
    rides_between["Day of Week"] = rides_between["pickup_dt"].dt.dayofweek
    rides_between["Hour"] = rides_between["pickup_dt"].dt.hour

    duration_table = (
        pd.pivot_table(rides_between, "Duration", "Day of Week", "Hour")
        .loc[:, range(6, 24, 4)]
        .apply(lambda x: pd.to_timedelta(x, unit="s").dt.round("min"))
        .rename(dict(enumerate(calendar.day_name)))
    )

    between_map = folium.Map(location=NYC_CENTER, tiles=TILES, zoom_start=12)

    folium.Marker(
        pickup_coords, icon=folium.Icon(color="red", icon="taxi", prefix="fa")
    ).add_to(between_map)

    folium.Marker(
        dropoff_coords,
        icon=folium.Icon(color="green", icon="flag-checkered", prefix="fa"),
    ).add_to(between_map)

    return duration_table, between_map


def find_closest_rides(rides_df, poi_address, mode="both", radius=10):

    assert mode in ("both", "pickup", "dropoff")

    expanded_modes = ("pickup", "dropoff") if mode == "both" else (mode,)

    poi_location = geolocator.geocode(poi_address)
    poi_coords = (poi_location.latitude, poi_location.longitude)

    closest_rides_dfs = {}
    for m in expanded_modes:
        dists = distance_poi(rides_df, poi_coords, mode=m)
        closest_mask = dists < radius
        closest_rides_dfs[m] = rides_df[closest_mask]

    return closest_rides_dfs, poi_coords


def plot_markers(df):

    map_ = folium.Map(location=NYC_CENTER, tiles=TILES, zoom_start=12)

    for _, row in df.iterrows():

        try:
            marker_address = geolocator.reverse((row["lat"], row["lng"])).address.rsplit(
                ",", 4
            )[0]
        except GeocoderTimedOut:
            marker_address = ""
        
        maker_popoutf = (
            f"<b>Address:</b> {marker_address}\n" f'<b>Timestamp:</b> {row["dt"]}'
        )

        icon = (
            folium.Icon(color="red", icon="taxi", prefix="fa")
            if row["type_"] == "pickup"
            else folium.Icon(color="green", icon="flag-checkered", prefix="fa")
        )

        folium.Marker([row["lat"], row["lng"]], popup=maker_popoutf, icon=icon).add_to(
            map_
        )

    return map_


def plot_closest_rides(rides_df, poi_address, mode="both", radius=10):

    closest_rides_dfs, poi_coords = find_closest_rides(
        rides_df, poi_address, mode, radius
    )

    dfs = []

    for m, mode_closest_rides_df in closest_rides_dfs.items():
        opposite_mode = "pickup" if m == "dropoff" else "dropoff"
        dfs.append(
            mode_closest_rides_df[
                [f"{opposite_mode}_lat", f"{opposite_mode}_lng", f"{opposite_mode}_dt"]
            ]
            .rename(
                {
                    f"{opposite_mode}_lat": "lat",
                    f"{opposite_mode}_lng": "lng",
                    f"{opposite_mode}_dt": "dt",
                },
                axis=1,
            )
            .assign(type_=opposite_mode)
        )

    map_ = plot_markers(pd.concat(dfs))

    folium.Circle(
        radius=radius,
        location=poi_coords,
        popup=poi_address,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
    ).add_to(map_)

    return map_


def plot_grid_map(cell_size):
    grid_map = folium.Map(location=NYC_CENTER, tiles=TILES, zoom_start=12)

    for shift in it.product(np.arange(-0.1, 0.1, cell_size), repeat=2):
        shift_arr = np.array(shift)
        bounds = [NYC_CENTER + shift_arr, NYC_CENTER + shift_arr + cell_size]

        folium.Rectangle(
            bounds=bounds,
            color="#ff7800",
            fill=True,
            fill_color="#ffff00",
            fill_opacity=0.02,
        ).add_to(grid_map)

    return grid_map
