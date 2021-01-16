import face_recognition
import requests

#这里的链接你可以改成网上任意的照片链接，只要保证后缀是.jpg或.png计算机可读取即可
response = requests.get("https://raw.githubusercontent.com/neowalter/Neural_network/main/pics/MUSK.jpg")
file = open("MUSK.jpg", "wb")
file.write(response.content)
file.close()

response = requests.get("https://raw.githubusercontent.com/neowalter/Neural_network/main/pics/MHT.jpg")
file = open("MHT.jpg", "wb")
file.write(response.content)
file.close()

response = requests.get("https://raw.githubusercontent.com/neowalter/Neural_network/main/pics/Elon-Musk-2010.jpg")
file = open("p3.jpg", "wb")
file.write(response.content)
file.close()

MUSK = face_recognition.load_image_file("MUSK.jpg")
MHT = face_recognition.load_image_file("MHT.jpg")

# Get the face encoding of each person. This can fail if no one is found in the photo.
musk_face_encoding = face_recognition.face_encodings(MUSK)[0]
mht_face_encoding = face_recognition.face_encodings(MHT)[0]

# Create a list of all known face encodings
known_face_encodings = [
    musk_face_encoding,
    mht_face_encoding,
]


# Load the image we want to check
unknown_image = face_recognition.load_image_file("p3.jpg")

# Get face encodings for any people in the picture
face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

    name = "Unknown"

    if results[0]:
        name = "MUSK"

    elif results[1]:
        name = "MHT"

    print(f"Found {name} in the photo!")
