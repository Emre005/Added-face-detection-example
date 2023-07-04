## Face Detection

Bu örnekte, OpenCV kullanarak bir video dosyası üzerinde yüz algılama yapılıyor. Kullandığımız yüz algılama modeli, OpenCV'nin `haarcascade_frontalface_default.xml` dosyasında bulunan bir Haar cascade'dir.

Kodun çalışması için `haarcascade_frontalface_default.xml` dosyasının doğru yolda olması gerekmektedir. Bu dosya, OpenCV'nin GitHub reposundan indirilebilir.

### Kullanım

Aşağıdaki şekilde çalıştırabilirsiniz:

python face_detection.py

Bu komut, `video.mp4` isimli video dosyası üzerinde yüz algılama yapar. Algılanan yüzlerin etrafında bir dikdörtgen çizer ve sonuçları ekrana gösterir.
