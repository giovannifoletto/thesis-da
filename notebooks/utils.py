#!/usr/bin/env
import json

FLAWS_CLOUDTRAILS_FILES = [
    "../data/raw/flaws_cloudtrail00.json",
    "../data/raw/flaws_cloudtrail02.json",
    "../data/raw/flaws_cloudtrail03.json",
    "../data/raw/flaws_cloudtrail04.json",
    "../data/raw/flaws_cloudtrail05.json",
    "../data/raw/flaws_cloudtrail06.json",
    "../data/raw/flaws_cloudtrail07.json",
    "../data/raw/flaws_cloudtrail08.json",
    "../data/raw/flaws_cloudtrail09.json",
    "../data/raw/flaws_cloudtrail10.json",
    "../data/raw/flaws_cloudtrail11.json",
    "../data/raw/flaws_cloudtrail12.json",
    "../data/raw/flaws_cloudtrail13.json",
    "../data/raw/flaws_cloudtrail14.json",
    "../data/raw/flaws_cloudtrail15.json",
    "../data/raw/flaws_cloudtrail16.json",
    "../data/raw/flaws_cloudtrail17.json",
    "../data/raw/flaws_cloudtrail18.json",
    "../data/raw/flaws_cloudtrail19.json"
]

def dowload_original_files():

    # Deprecated, using DVC instead.
    import requests
    FLAWS_LOG_URL = "https://summitroute.com/downloads/flaws_cloudtrail_logs.tar"
    r = requests.get(FLAWS_LOG_URL, stream=True)
    print(r.status_code)
    #print(r.raw.read()[:1000])
    if r.status_code == 200:
        with open("flaws_cloudtrails_log.tar", 'wb') as f:
            f.write(r.raw.read())

def untar_files():
    import os

    os.system("tar -xvf ../data/raw/flaws_cloudtrails_log.tar -C data/raw/")
    os.system("gzip -d ../data/raw/flaws_cloudtrail_logs/*")
    os.system("mv ../data/raw/flaws_cloudtrail_logs/* ../data/raw/")


def import_data():
    records = []

    for i in FLAWS_CLOUDTRAILS_FILES[1:3]:
        file_open = open(i, "r")
        line = file_open.readline()
        jo = json.loads(line)
        for ji in jo["Records"]:
            records.append(ji)      
        file_open.close()
        return records
    
if __name__ == "__main__":
    import_data()