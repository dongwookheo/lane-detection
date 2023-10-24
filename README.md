# lane-detection
## 환경 구축
### 0. git clone
```
git clone https://github.com/dongwookheo/lane-detection.git
```

### 1. Install OpenCV
```
cd lane-detection/thirdparty/OpenCV/
```
- ../lane-detection/thirdparty/OpenCV/ 폴더 내에서
```
git clone https://github.com/opencv/opencv.git
```

### 2. Build thrid party
```
cd build/
```
- ../lane-detection/thirdparty/OpenCV/build 폴더 내에서
```
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install ../opencv
```
- Makefile 생성 완료 후, ```nproc``` 명령으로 <core_num>을 확인하고 아래의 명령 (다른 작업과 병행 시, 1~2 적게)
```
make -j<core_num>
```
```
make install
```
