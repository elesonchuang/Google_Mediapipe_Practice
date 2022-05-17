import os
import os.path as osp
import sys
BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)
import argparse

import grpc
import fib_pb2
import fib_pb2_grpc


def main(args):
    host = f"{args['ip']}:{args['port']}"
    print(host)
    with grpc.insecure_channel(host) as channel:
        stub = fib_pb2_grpc.FibCalculatorStub(channel)

        request = fib_pb2.FibRequest()
        request.order = args['order']

        response = stub.Compute(request)
        print(response.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--order", type=int, default=1)
    args = vars(parser.parse_args())
    main(args)
    while True:
        input_txt = input("Choose operate mode as you wish...")
        if input_txt not in ("1", "2", "3", "exit"):
            print("""Mode: 1 => Object Detection
                           2 => Hand Pose Tracking
                           3 => Pose Estimation
                           exit => Stop and leave this program""")
            continue
        else:
            if input_txt == "exit":break
            args["order"] = int(input_txt)
            main(args)

