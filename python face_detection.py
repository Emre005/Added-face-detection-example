import cv2

# Yüz tanıma modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video dosyasını oku
cap = cv2.VideoCapture('video.mp4')  # Buraya video dosyanızın yolunu yazın

while True:
    # Video akışından bir frame al
    ret, img = cap.read() 

    # Frame'i gri tonlamaya dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Gri tonlamalı frame'de yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Her algılanan yüz için bir dikdörtgen çiz
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # Sonuç frame'ini göster
    cv2.imshow('img',img)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
