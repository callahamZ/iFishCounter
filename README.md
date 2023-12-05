# iFishCounter
Dibuat oleh M Ilham Kurniawan

Wirausaha Merdeka PENS 2023


### Pengenalan
Ini adalah program penghitung ikan yang menggunakan algoritma deteksi YOLOV8.


Program utama ada pada file
> fishCounter Video.py


File `sort.py` adalah algoritma untuk mendeteksi pergerakan objek sehingga apabila objek berpindah tempat, tidak dihitung sebagai objek baru.
Source : [abewley/sort](https://github.com/abewley/sort)


### Cara kerja

Program ini menggunakan **cv2** untuk membaca setiap frame dari video yang di masukkan. Kemudian di masukkan ke dalam model AI menjadi variable `img`. Sehingga pada dasarnya algoritma ini mengolah deteksi satu gambar setiap saat dengan kecepatan sesuai spesifikasi perangkat user.

## Training

Contoh [dataset yang saya gunakan](https://app.roboflow.com/ds/C87Ut9iz9y?key=8sJSksjRZy)


Sebelum menjalankan program training, ubah filepath pada `data.yaml` ke _path absolute_ dari dataset yang tersimpan di komputer anda. Sesuaikan nilai `epochs` sesuai keinginan. Setelah training selesai, akan muncul folder **runs** yang berisi hasil training sebelumnya. Model bernama `last.pt` dan `best.pt` akan muncul pada sebuah folder dan dapat digunakan untuk dijalankan pada program utama.


Selengkapnya bisa lihat dokumentasi [YOLOV8](https://github.com/ultralytics/ultralytics)


## Pengembangan

Untuk mengembangkan program ini untuk di jalankan menggunakan web atau aplikasi smartphone, dapat dengan menjalankan program ini dan mengirimkan frame atau gambar yang ingin di deteksi untuk di masukkan menjadi variable `img`. Sehingga perlu modifikasi untuk mengubah sistem program ini menjadi sebuah **fungsi**.
