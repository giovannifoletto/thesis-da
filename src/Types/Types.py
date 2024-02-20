
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
    "eventVersion": pl.Utf8,
    "userIdentity": pl.Struct({
        "type": pl.Utf8,
        "principalId": pl.Utf8,
        "arn": pl.Utf8,
        "accountId": pl.Utf8,
        "invokedby": pl.Utf8,
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
    "requestId": pl.Utf8,
    "eventId": pl.Utf8,
    "resources": pl.List(pl.Struct({
        "ARN": pl.Utf8,
        "accountId": pl.Utf8,
        "type": pl.Utf8,
    })),
    "eventType": pl.Utf8,
    "apiVersion": pl.Utf8,
    "readonly": pl.Utf8,
    "recipientAccountId": pl.Utf8,
    "serviceEventDetails": pl.Utf8,
    "sharedEventId": pl.Utf8,
    "vpcEndpointId": pl.Utf8,
}