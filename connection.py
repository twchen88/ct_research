from sshtunnel import SSHTunnelForwarder
import pymysql

def connect():
    server = SSHTunnelForwarder(
    ssh_address=('54.146.149.17', 22),
    ssh_username="ec2-user",
    ssh_pkey="/home/chloe/Research/buaws-kirans-ec2.pem",
    remote_bind_address=("ctprod.cktdvwrzgusc.us-east-1.rds.amazonaws.com", 3306)
    )

    server.start()

    with open("credentials.txt", "r") as f:
        lines = f.read().splitlines()
        user = lines[0]
        pw = lines[1]

    con = pymysql.connect(user=user,passwd=pw,db='constant_therapy',host='127.0.0.1',port=server.local_bind_port)

    return con