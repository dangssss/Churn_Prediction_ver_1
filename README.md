# Churn.Deployment

## Build image từ Dockerfile
```docker build -t dockerhub.vnpost.vn/airflow-churn:1.0.0 .```

> Lưu ý version phải tăng dần. Ví dụ: 1.0.1, 1.0.2, 1.1.0, 2.0.0... Trùng version sẽ bị lỗi

## Đẩy lên docker hub của Tổng công ty
```docker push dockerhub.vnpost.vn/airflow-churn:1.0.0```

> Lưu ý 1: phải có tài khoản dockerhub.vnpost.vn mới có thể đẩy được

> Lưu ý 2: Xin tài khoản liên hệ thanhnt@vnpost.vn, SĐT: +84961186043

> Lưu ý 3: đăng nhập vào dockerhub.vnpost.vn bằng lệnh:

 ```docker login dockerhub.vnpost.vn -u username -p password```

 ## Cấu hình vào docker-compose.yaml
 Tạo file .env

 Thêm biến: 

```AIRFLOW_IMAGE_NAME=dockerhub.vnpost.vn/airflow-churn:1.0.0```
...
