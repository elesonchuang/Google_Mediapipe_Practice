# gRPC-with-protobuf

## How to run
- Test on Nvidia Jetson Nano 2GB Developer Kit with PiCamera V2
- Install project dependencies
```bash
# Install protobuf compiler
$ sudo apt-get install protobuf-compiler

# Install buildtools
$ sudo apt-get install build-essential make

# Install grpc packages
$ pip3 install -r requirements.txt
```
- Compile protobuf schema to python wrapper
```bash
$ make
```
- Start the gRPC service
```bash
$ python3 server.py 
```
- Start the gRPC client
```bash
$ python3 client.py 
```
  - You should see
  ```bash
  Choose operate mode as you wish...
  ```
  Type 1 | 2 | 3 to choose mode "object" | "hand" | "pose" or type "exit" to leave.
