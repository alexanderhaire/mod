
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

def test_distance():
    print("Testing Geopy...")
    try:
        geolocator = Nominatim(user_agent="cdi_test_script_v2")
        
        # Test 1: Simple City
        print("Geocoding 'Tampa, FL'...")
        loc1 = geolocator.geocode("Tampa, FL", timeout=5)
        if loc1:
            print(f"Found Tampa: {loc1.latitude}, {loc1.longitude}")
        else:
            print("Failed to find Tampa")

        # Test 2: User Address (if possible, or just re-use Tampa for distance)
        # 4206 Business Ln, Plant City, FL (Fixed Dest)
        dest_coords = (28.0186, -82.1129)
        
        if loc1:
            origin_coords = (loc1.latitude, loc1.longitude)
            dist = geodesic(origin_coords, dest_coords).miles
            print(f"Distance from Tampa to Plant City: {dist:.2f} miles")
            assert dist > 0
            print("PASS: Distance calculation works")
        
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_distance()
