
from geopy.geocoders import Nominatim

def get_coords():
    # User agent is required by Nominatim
    geolocator = Nominatim(user_agent="cdi_vendor_portal_test")
    location = geolocator.geocode("4206 Business Ln, Plant City, FL")
    if location:
        print(f"LAT={location.latitude}")
        print(f"LON={location.longitude}")
    else:
        print("Address not found")

if __name__ == "__main__":
    get_coords()
