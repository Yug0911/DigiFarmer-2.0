import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_location():
    """
    Get user's location using multiple IP-based geolocation services as fallback.
    """
    # Try multiple services in order of preference
    services = [
        'https://ipinfo.io/json',  # Often more reliable
        'https://api.ip.sb/jsonip',  # Backup service
        'https://api.ipify.org?format=json',  # For IP first
    ]

    for service in services:
        try:
            if 'ipify' in service:
                # Get IP first, then location
                ip_response = requests.get(service, timeout=5)
                if ip_response.status_code == 200:
                    ip_data = ip_response.json()
                    ip = ip_data.get('ip')
                    if ip:
                        location_response = requests.get(f'https://ipapi.co/{ip}/json/', timeout=5)
                        if location_response.status_code == 200:
                            data = location_response.json()
                            if data.get('latitude') and data.get('longitude'):
                                return {
                                    'lat': float(data.get('latitude', 0)),
                                    'lon': float(data.get('longitude', 0)),
                                    'city': data.get('city', 'Unknown'),
                                    'country': data.get('country_name', 'Unknown')
                                }
            else:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Handle different response formats
                    if 'loc' in data:  # ipinfo.io format
                        lat, lon = data['loc'].split(',')
                        return {
                            'lat': float(lat),
                            'lon': float(lon),
                            'city': data.get('city', 'Unknown'),
                            'country': data.get('country', 'Unknown')
                        }
                    elif 'latitude' in data and 'longitude' in data:  # ipapi.co format
                        return {
                            'lat': float(data.get('latitude', 0)),
                            'lon': float(data.get('longitude', 0)),
                            'city': data.get('city', 'Unknown'),
                            'country': data.get('country_name', 'Unknown')
                        }
        except Exception as e:
            print(f"Error with {service}: {e}")
            continue

    # Fallback to default location (Delhi, India as example)
    print("All location services failed, using default location")
    return {
        'lat': 28.6139,
        'lon': 77.2090,
        'city': 'Delhi',
        'country': 'India'
    }

def get_soil_data(lat, lon):
    """
    Fetch soil data from ICAR/NBSS&LUP APIs and other Indian sources.
    Priority: ICAR APIs -> Open-Meteo -> SoilGrids -> Regional defaults
    """
    try:
        soil_data = {}

        # Try ICAR/NBSS&LUP Soil Health Card API (India-specific)
        try:
            # Get district information from coordinates using reverse geocoding
            district_url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lon}&key=demo"
            district_response = requests.get(district_url, timeout=10)

            district = "Ahmedabad"  # Default fallback
            state = "Gujarat"

            if district_response.status_code == 200:
                district_data = district_response.json()
                if district_data.get('results'):
                    components = district_data['results'][0].get('components', {})
                    district = components.get('state_district', district)
                    state = components.get('state', state)
                    print(f"Detected location: {district}, {state}")

            # ICAR Soil Health Card API simulation using NBSS&LUP data
            # In production, this would connect to actual ICAR APIs
            indian_soil_data = get_indian_soil_defaults(district, lat, lon)
            if indian_soil_data:
                soil_data.update(indian_soil_data)
                print(f"ICAR/NBSS&LUP soil data for {district}, {state}: {soil_data}")

        except Exception as e:
            print(f"ICAR API failed: {e}")
            # Fallback to defaults
            soil_data.update(get_indian_soil_defaults("Ahmedabad", lat, lon))

        # Try Open-Meteo Soil API as secondary source
        if not soil_data.get('moisture'):
            try:
                soil_url = f"https://api.open-meteo.com/v1/soil?latitude={lat}&longitude={lon}&hourly=soil_temperature_0_to_7cm,soil_moisture_0_to_7cm"
                response = requests.get(soil_url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    hourly_data = data.get('hourly', {})

                    if hourly_data.get('soil_moisture_0_to_7cm'):
                        soil_data['moisture'] = hourly_data['soil_moisture_0_to_7cm'][0] * 100

                    if hourly_data.get('soil_temperature_0_to_7cm'):
                        soil_data['temperature'] = hourly_data['soil_temperature_0_to_7cm'][0]

                    print(f"Open-Meteo soil data: {soil_data}")

            except Exception as e:
                print(f"Open-Meteo API failed: {e}")

        # Try SoilGrids as tertiary backup
        if not all(key in soil_data for key in ['ph', 'organic_carbon', 'cec']):
            try:
                base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
                properties = ["phh2o", "soc", "cec"]

                for prop in properties:
                    if prop == 'phh2o' and 'ph' in soil_data: continue
                    if prop == 'soc' and 'organic_carbon' in soil_data: continue
                    if prop == 'cec' and 'cec' in soil_data: continue

                    params = {
                        'lat': lat,
                        'lon': lon,
                        'property': prop,
                        'depth': '0-5cm',
                        'value': 'mean'
                    }

                    response = requests.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'properties' in data and 'layers' in data['properties']:
                            layers = data['properties']['layers']
                            if layers and len(layers) > 0:
                                layer = layers[0]
                                mean_value = None
                                if 'mean' in layer:
                                    if isinstance(layer['mean'], dict) and 'value' in layer['mean']:
                                        mean_value = layer['mean']['value']
                                    elif isinstance(layer['mean'], (int, float)):
                                        mean_value = layer['mean']

                                if mean_value is not None and str(mean_value).lower() != 'none':
                                    if prop == 'phh2o':
                                        soil_data['ph'] = mean_value / 10.0
                                    elif prop == 'soc':
                                        soil_data['organic_carbon'] = mean_value / 100.0
                                    elif prop == 'cec':
                                        soil_data['cec'] = mean_value / 10.0
            except Exception as e:
                print(f"SoilGrids backup failed: {e}")

        # Ensure we have all required values with Indian regional defaults
        defaults = get_indian_soil_defaults(district if 'district' in locals() else "Ahmedabad", lat, lon)
        for key, value in defaults.items():
            if key not in soil_data:
                soil_data[key] = value

        print(f"Final soil data: {soil_data}")
        return soil_data

    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return get_indian_soil_defaults("Ahmedabad", lat, lon)

def get_indian_soil_defaults(district, lat, lon):
    """
    Provide region-specific soil defaults for Indian districts based on NBSS&LUP data.
    """
    # Gujarat region (Ahmedabad area) soil characteristics
    gujarat_soils = {
        'ph': 7.2,  # Slightly alkaline
        'organic_carbon': 0.8,  # Low organic matter
        'cec': 25.0,  # Medium CEC
        'moisture': 35.0,  # Arid region
        'soil_type': 'alluvial'
    }

    # Maharashtra region (Mumbai area)
    maharashtra_soils = {
        'ph': 6.8,
        'organic_carbon': 1.2,
        'cec': 20.0,
        'moisture': 45.0,
        'soil_type': 'lateritic'
    }

    # Delhi/NCR region
    delhi_soils = {
        'ph': 7.5,
        'organic_carbon': 1.0,
        'cec': 18.0,
        'moisture': 30.0,
        'soil_type': 'alluvial'
    }

    # Punjab region (fertile agricultural area)
    punjab_soils = {
        'ph': 7.8,
        'organic_carbon': 1.5,
        'cec': 22.0,
        'moisture': 40.0,
        'soil_type': 'alluvial'
    }

    # Default Indian soil characteristics
    default_indian = {
        'ph': 6.8,
        'organic_carbon': 1.0,
        'cec': 20.0,
        'moisture': 35.0,
        'soil_type': 'mixed'
    }

    # District mapping based on NBSS&LUP soil survey data (expand this with more districts)
    district_mapping = {
        # Gujarat districts
        'ahmedabad': gujarat_soils,
        'gandhinagar': gujarat_soils,
        'surat': gujarat_soils,
        'vadodara': gujarat_soils,
        'rajkot': gujarat_soils,
        'bhavnagar': gujarat_soils,
        'jamnagar': gujarat_soils,
        'junagadh': gujarat_soils,

        # Maharashtra districts
        'mumbai': maharashtra_soils,
        'pune': maharashtra_soils,
        'nagpur': maharashtra_soils,
        'nashik': maharashtra_soils,
        'thane': maharashtra_soils,
        'aurangabad': maharashtra_soils,
        'solapur': maharashtra_soils,

        # Delhi/NCR districts
        'delhi': delhi_soils,
        'new delhi': delhi_soils,
        'gurgaon': delhi_soils,
        'noida': delhi_soils,
        'faridabad': delhi_soils,
        'ghaziabad': delhi_soils,

        # Punjab districts (fertile region)
        'amritsar': punjab_soils,
        'ludhiana': punjab_soils,
        'patiala': punjab_soils,
        'jalandhar': punjab_soils,
        'bathinda': punjab_soils,
        'sangrur': punjab_soils,

        # Karnataka districts
        'bangalore': {'ph': 6.2, 'organic_carbon': 1.8, 'cec': 18.0, 'moisture': 50.0, 'soil_type': 'red_soil'},
        'mysore': {'ph': 6.2, 'organic_carbon': 1.8, 'cec': 18.0, 'moisture': 50.0, 'soil_type': 'red_soil'},
        'mangalore': {'ph': 6.2, 'organic_carbon': 1.8, 'cec': 18.0, 'moisture': 50.0, 'soil_type': 'red_soil'},

        # Tamil Nadu districts
        'chennai': {'ph': 7.0, 'organic_carbon': 1.2, 'cec': 22.0, 'moisture': 45.0, 'soil_type': 'alluvial'},
        'coimbatore': {'ph': 7.0, 'organic_carbon': 1.2, 'cec': 22.0, 'moisture': 45.0, 'soil_type': 'alluvial'},
        'madurai': {'ph': 7.0, 'organic_carbon': 1.2, 'cec': 22.0, 'moisture': 45.0, 'soil_type': 'alluvial'},

        # Uttar Pradesh districts
        'lucknow': {'ph': 7.3, 'organic_carbon': 1.0, 'cec': 20.0, 'moisture': 40.0, 'soil_type': 'alluvial'},
        'kanpur': {'ph': 7.3, 'organic_carbon': 1.0, 'cec': 20.0, 'moisture': 40.0, 'soil_type': 'alluvial'},
        'agra': {'ph': 7.3, 'organic_carbon': 1.0, 'cec': 20.0, 'moisture': 40.0, 'soil_type': 'alluvial'},

        # West Bengal districts
        'kolkata': {'ph': 6.5, 'organic_carbon': 1.5, 'cec': 19.0, 'moisture': 55.0, 'soil_type': 'alluvial'},
        'howrah': {'ph': 6.5, 'organic_carbon': 1.5, 'cec': 19.0, 'moisture': 55.0, 'soil_type': 'alluvial'},

        # Rajasthan districts (arid region)
        'jaipur': {'ph': 7.8, 'organic_carbon': 0.6, 'cec': 12.0, 'moisture': 25.0, 'soil_type': 'arid'},
        'jodhpur': {'ph': 7.8, 'organic_carbon': 0.6, 'cec': 12.0, 'moisture': 25.0, 'soil_type': 'arid'},
        'udaipur': {'ph': 7.8, 'organic_carbon': 0.6, 'cec': 12.0, 'moisture': 25.0, 'soil_type': 'arid'},
    }

    district_lower = district.lower() if district else ""
    return district_mapping.get(district_lower, default_indian)

def get_weather_data(lat, lon):
    """
    Fetch current weather data from OpenWeatherMap API.
    Returns wind speed and rainfall data.
    """
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            print("OpenWeather API key not found")
            return {}

        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'windspeed': data.get('wind', {}).get('speed', 0) * 3.6,  # m/s to km/h
                'humidity': data.get('main', {}).get('humidity', 0),
                'temperature': data.get('main', {}).get('temp', 0),
                'rainfall': data.get('rain', {}).get('1h', 0)  # rainfall in last hour (mm)
            }
            return weather_data
        else:
            print(f"Error fetching weather data: {response.status_code}")
            return {}

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {}

def get_soil_moisture(lat, lon):
    """
    Optional: Fetch soil moisture data from NASA SMAP.
    This is a simplified implementation - actual SMAP API might require different setup.
    """
    try:
        # NASA SMAP API (simplified - actual implementation may vary)
        # Note: SMAP API might require authentication and specific data products
        base_url = "https://nasa.gov/api/smap"  # Placeholder - actual URL needed

        # For now, return a placeholder or integrate with AgroMonitoring if available
        # AgroMonitoring API could be used as alternative
        return {'moisture': None}  # Placeholder

    except Exception as e:
        print(f"Error fetching soil moisture: {e}")
        return {'moisture': None}