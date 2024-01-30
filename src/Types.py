
# Giovanni Foletto - CloudTrailLog Template
from dataclasses import dataclass

@dataclass
class userIdentity:
    type:    str
    principalid: str
    arn:         str
    accountid:   str
    invokedby:   str
    accesskeyid: str
    userName:    str

@dataclass
class attributes:
    mfaauthenticated: str
    creationdate:     str

@dataclass
class sessionissuer:
    type:    str
    principalId: str
    arn:         str
    accountId:   str
    userName:    str


@dataclass
class sessioncontext:
    attributes: attributes
    sessionissuer: sessionissuer

@dataclass
class resources:
    ARN: str
    accountId: str
    type: str

@dataclass
class CloudtrailLog:
    eventversion: str
    useridentity: userIdentity
    sessioncontext: sessioncontext
    eventtime: str
    eventsource: str
    eventname: str
    awsregion: str
    sourceipaddress: str
    useragent: str
    errorcode: str
    errormessage: str
    requestparameters: str
    responseelements: str
    additionaleventdata: str
    requestid: str
    eventid: str
    resources: [resources]
    eventtype: str
    apiversion: str
    readonly: str
    recipientaccountid: str
    serviceeventdetails: str
    sharedeventid: str
    vpcendpointid: str


import polars as pl
cloudtrails_log_schema = {
    "eventversion": pl.Utf8,
    "useridentity": {
        "type": pl.Utf8,
        "principalid": pl.Utf8,
        "arn": pl.Utf8,
        "accountid": pl.Utf8,
        "invokedby": pl.Utf8,
        "accesskeyid": pl.Utf8,
        "userName": pl.Utf8,
    },
    "sessioncontext": {
        "attributes": {
            "mfaauthenticated": pl.Utf8,
            "creationdate": pl.Utf8,
        },
        "sessionissuer": {
            "type": pl.Utf8,
            "principalId": pl.Utf8,
            "arn": pl.Utf8,
            "accountId": pl.Utf8,
            "userName": pl.Utf8,
        },
    },
    "eventtime": pl.Utf8,
    "eventsource": pl.Utf8,
    "eventname": pl.Utf8,
    "awsregion": pl.Utf8,
    "sourceipaddress": pl.Utf8,
    "useragent": pl.Utf8,
    "errorcode": pl.Utf8,
    "errormessage": pl.Utf8,
    "requestparameters": pl.Utf8,
    "responseelements": pl.Utf8,
    "additionaleventdata": pl.Utf8,
    "requestid": pl.Utf8,
    "eventid": pl.Utf8,
    "resources": pl.List({
        "ARN": pl.Utf8,
        "accountId": pl.Utf8,
        "type": pl.Utf8,
    }),
    "eventtype": pl.Utf8,
    "apiversion": pl.Utf8,
    "readonly": pl.Utf8,
    "recipientaccountid": pl.Utf8,
    "serviceeventdetails": pl.Utf8,
    "sharedeventid": pl.Utf8,
    "vpcendpointid": pl.Utf8,
}