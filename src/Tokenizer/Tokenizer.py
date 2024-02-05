#!/usr/bin/env python3
import numpy as np
import polars as pl
import urllib.parse
import gc
#from ..Types.Types import cloudtrails_log_schema

cloudtrails_log_schema = {
    "eventVersion": pl.Float64,
    "userIdentity": pl.Struct({
        "type": pl.Utf8,
        "principalId": pl.Utf8,
        "arn": pl.Utf8,
        "accountId": pl.Utf8,
        "invokedBy": pl.Utf8,
        "accesskeyId": pl.Utf8,
        "userName": pl.Utf8,
    }),
    "sessionContext": pl.Struct({
        "attributes": pl.Struct({
            "mfaAuthenticated": pl.Utf8,
            "creationDate": pl.Utf8,
        }),
        "sessionIssuer": pl.Struct({
            "type": pl.Utf8,
            "principalId": pl.Utf8,
            "arn": pl.Utf8,
            "accountId": pl.Utf8,
            "userName": pl.Utf8,
        }),
    }),
    "eventTime": pl.Utf8,
    "eventSource": pl.Utf8,
    "eventName": pl.Utf8,
    "awsRegion": pl.Utf8,
    "sourceIPAddress": pl.Utf8,
    "userAgent": pl.Utf8,
    "errorCode": pl.Utf8,
    "errorMessage": pl.Utf8,
    "requestParameters": pl.Utf8,
    "responseElements": pl.Utf8,
    "additionalEventData": pl.Utf8,
    "requestID": pl.Utf8,
    "eventID": pl.Utf8,
    "resources": pl.List(pl.Struct({
        "arn": pl.Utf8,
        "accountId": pl.Utf8,
        "type": pl.Utf8,
    })),
    "eventType": pl.Utf8,
    "apiVersion": pl.Utf8,
    "readOnly": pl.Utf8,
    "recipientAccountId": pl.Int64,
    "serviceEventDetails": pl.Utf8,
    "sharedEventId": pl.Utf8,
    "vpcEndpointId": pl.Utf8,
}

from enum import Enum
class AWSKubeLog(Enum):
    userAgent: str
#    eventID: str
    eventType: str
    sourceIpAddress: str
    eventName: str
    eventSource: str
    recipientAccountId: str
    awsRegion: str
#    requestID: str
    eventVersion: str
#    eventTime: str
    errorMessage: str
    errorCode: str
#    readOnly: str

TO_BE_IMPORTED = [
    "userAgent",
    "eventType",
    #"sourceIpAddress",
    "eventName",
    "eventSource",
    "recipientAccountId",
    "awsRegion",
    "eventVersion", # TO INVESTIGATE WHY IS LIKE THIS
    #"errorMessage",
    "errorCode",
]

class Tokenizer:
    def __init__(self, path):
        self.dlp    = pl.read_csv(path, separator="|")
        self.columns= dict()

    def create_vocabulary(self):
        for col in TO_BE_IMPORTED:
            self.columns[col] = self.dlp.select(pl.col(col)).unique().to_numpy()
            print(f"COL: {col}, {self.columns[col].shape}")

    def get_token(self, value):
        # res = self.columns[value].index(1)
        res = []
        if len(self.columns[value]) <= 0:
            print(self.columns[value])
            return []
        for col in self.columns[value]:
            try:
            # print(col, np.where(self.columns[value] == col)[0][0])
            # break
                insert_value = np.where(self.columns[value] == col)
                res.append([col[0], insert_value[0][0]+1])
            except:
                print(insert_value)

        print(f"Col: {len(res)}")

        return res

    def main(self):
        self.create_vocabulary()
        for col in self.columns:
            print(f"Working on {col}")
            to_join = pl.DataFrame(self.get_token(col), schema={col: pl.Utf8, f"{col}-token": pl.Int64})
            print(to_join.dtypes, to_join.shape)

            to_join.write_csv(f"../../data/prepared/vocabularies/{col}.csv")

        del to_join
        del self.dlp
        del self.columns
        gc.collect()

        print("-- Making Joins --")
        dlp = pl.read_ndjson("../../data/raw/flaws_cloudtrail00.ndjson", schema=cloudtrails_log_schema)
        # dlp = dlp.select(
        #         pl.col(
        #             "userAgent",
        #             "eventType",
        #             "sourceIpAddress",
        #             "eventName",
        #             "eventSource",
        #             "recipientAccountId",
        #             "awsRegion",
        #             "eventVersion",
        #             "errorCode",
        #         )
        #     )
        print(f"DLP-SHAPE: {dlp.shape}, {dlp.columns}")

        for col in TO_BE_IMPORTED:
            print(f"COL: {col}")
            dlp_1 = pl.read_csv(f"../../data/prepared/vocabularies/{col}.csv")
            dlp = dlp.join(dlp_1, on=col)
            del dlp_1
            gc.collect()

        dlp.write_ndjson("../../data/prepared/all_vocabularies_riunited.ndjson")

# def getTokenByWord():
#     import json

#     word = dict()
#     file_obj = open("../data/not_fixed_analysis.json", "r").readlines()
#     for js in file_obj:
#         jo = json.loads(js)
#         #parser = Parser(jo)
#         for v in jo.values(): # parser.element:
#             print([a.split(" ") for a in v])
#         break

if __name__ == "__main__":

    # awskubelog = AWSKubeLog()
    # print([e.value for e in awskubelog])
    path = "../../data/prepared/fixed.csv"
    print(f"Name: {__name__}")
    print(f"Package: {__package__}")
    tokenizer = Tokenizer(path)
    tokenizer.main()
    #getTokenByWord()
