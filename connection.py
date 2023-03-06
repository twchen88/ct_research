### Create MySQL connection

from sshtunnel import SSHTunnelForwarder
import pymysql

def connect():

    # load in credential file
    with open("credentials.txt", "r") as f:
        lines = f.read().splitlines()
        user = lines[0]
        pw = lines[1]
        ssh_address = lines[2]
        ssh_user = lines[3]
        key_path = lines[4]
        host_name = lines[5]

    # port forwarding
    server = SSHTunnelForwarder(
    ssh_address=(ssh_address, 22),
    ssh_username=ssh_user,
    ssh_pkey=key_path,
    remote_bind_address=(host_name, 3306)
    )

    server.start()

    # connection to MySQL
    con = pymysql.connect(user=user, passwd=pw, host='127.0.0.1', port=server.local_bind_port)

    print("Connection Successful")

    return con