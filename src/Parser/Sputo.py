#!/usr/bin/env python3

# Giovanni Foletto - Reimplementing Spell in order to get comprehension in
# a JSON compatible object.

import os
import json
import numpy as np
import urllib.parse
import string
from tqdm import tqdm
from dataclasses import dataclass

STRING = string.ascii_uppercase

class LogReader:
    def __init__(self, inlog) -> None:
        """
        inlog: give the JSON deserialized log object (it must be a dict())
        """
        assert type(inlog) == type(dict())

        self.inlog    = inlog
        self.keys     = set()
        self.element  = dict()

        self.unwrap(self.inlog)
    
    def unwrap(self, log, key="", context="") -> None:
        if type(log) == type(dict()):
            for k in log.keys():
                self.keys.add(k)
                if context == "":
                    self.unwrap(log[k], key=k, context=k)
                else:
                    self.unwrap(log[k], key=k, context=context+"."+k)
        elif type(log) == type(list()):
            for el in log:
                self.unwrap(el, key=key, context=context)
        # Technicly, "Records" field should not exists anymore
        else:
            if key == "assumeRolePolicyDocument":
                log = urllib.parse.unquote(log).replace("\n", "").replace(" ", "")
            
            self.element[context] = log

@dataclass
class LCSSputoContent:
    originalLog  : dict
    templateName : str
    templateID   : [str]
    keyName      : str
    counter      : int
    def __init__(self, originalLog="", templateName="", templateID=1, keyName="") -> None:
        self.originalLog    = originalLog   # contains the started seq1 log
        self.templateName   = templateName  # contains the template readable name
        self.templateID     = templateID    # contains the template id
        self.keyName        = keyName       # contains the key respect of the template is made
        self.counter        = 1             # how many log are of this type
    
    def __str__(self) -> str:
        return f"LCSSputoContent<originalLog={self.originalLog}, \
            templateName={self.templateName}, templateID={self.templateID}, \
            keyName={self.keyName}, counter={self.counter}>"

    def __repr__(self) -> str:
        return f"LCSSputoContent<originalLog={self.originalLog}, \
            templateName={self.templateName}, templateID={self.templateID}, \
            keyName={self.keyName}, counter={self.counter}>"

    def __gt__(self, other) -> bool:
        #print(f"t1: {self.templateName.split("-")}, t2: {other.templateName.split("-")}")
        t1 = self.templateName.split("-")[1]
        t2 = other.templateName.split("-")[1]
        if(t1>t2):
            return True
        else:
            return False

class LCSSputoLogFile:
    def __init__(self, originalLog="", templateName="", templateID=1, content="") -> None:
        self.originalLog    = originalLog   # contains the started seq1 log
        self.templateName   = templateName  # contains the template readable name
        self.templateID     = templateID    # contains the template id
        self.content        = content       # contains the list of different contents
    
    def __str__(self) -> str:
        return " ".join([a.templateName for a in self.content])

class SputuLogParser:
    def __init__(
            self, 
            indir="./", 
            outdir="./result/",
            filename=None
            # log_format=None, 
            # tau=0.5, 
            # rex=[], 
            # keep_para=True
        ) -> None:

        assert indir is not None and indir != ""
        assert filename is not None
        
        self.path      = indir
        self.outdir    = outdir
        self.filename  = filename
            
        self.log_templ = {"err": [], "req": [], "res": []}
        self.file = open(self.path + self.filename, "r")
        
        # self.tau       = tau
        # self.logformat = log_format
        # self.df_log    = None
        # self.rex       = rex
        # self.keep_para = keep_para
        
        if not os.path.exists(self.outdir): 
            os.makedirs(self.outdir)

    def __del__(self):
        self.file.close()

    def LCS(self, seq1, seq2):

        if type(seq1) != str:
            seq1 = str(seq1)
        
        if type(seq2) != str:
            seq2 = str(seq2)

        #print(f"SEQ1: {seq1}, SEQ2: {seq2}, => type(SEQ1) = {type(seq1)}, type(SEQ2) = {type(seq2)}")
        
        # inizializzo matrice con linee seq2 e colonne seq1
        lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]
                result.insert(0, seq1[lenOfSeq1 - 1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result
    
    def create_vocabulary(self, filter="Records", until=10):
        # main method to build the vocabulary of logs
        json_file = json.loads(self.file.read())

        array_log_1 = np.array(json_file[filter][:until])
        array_log_2 = np.array(json_file[filter][:until])

        for log_1 in tqdm(array_log_1):
            # save the temp array of LCS for every new array and
            # initialize parser class for log_1
            parser_1 = dict()
            parser_1["err"] = log_1.get("errorMessage")      if not None else "None"
            parser_1["res"] = log_1.get("responseElements")  if not None else "None"
            parser_1["req"] = log_1.get("requestParameters") if not None else "None"
        
            longest = dict()    # the array with most common chars        
            max_len = dict()    # the max len of the match
            longest["err"], longest["res"], longest["req"] = [], [], []
            max_len["err"], max_len["res"], max_len["req"] = 0, 0, 0
 
            for log_2 in array_log_2:
                parser_2 = dict()
                # check if the key exists on both classes
                parser_2["err"] = log_2.get("errorMessage")      if not None else "None"
                parser_2["res"] = log_2.get("responseElements")  if not None else "None"
                parser_2["req"] = log_2.get("requestParameters") if not None else "None"

                lcs_res = dict()
                lcs_res["err"] = self.LCS(parser_1["err"], parser_2["err"])
                lcs_res["res"] = self.LCS(parser_1["res"], parser_2["res"])
                lcs_res["req"] = self.LCS(parser_1["req"], parser_2["req"])

                if len(lcs_res["err"]) >= max_len["err"]:
                    longest["err"] = lcs_res["err"]
                    max_len["err"] = len(lcs_res["err"])
                
                if len(lcs_res["res"]) >= max_len["res"]:
                    longest["res"] = lcs_res["res"]
                    max_len["res"] = len(lcs_res["res"])
                
                if len(lcs_res["req"]) >= max_len["req"]:
                    longest["req"] = lcs_res["req"]
                    max_len["req"] = len(lcs_res["req"])
         
                find = dict()
                find["err"], find["req"], find["res"] = False, False, False

                for arr in self.log_templ["err"]:
                    if np.array_equiv(arr, lcs_res["err"]):
                        find["err"] = True

                for arr1 in self.log_templ["res"]:
                    if np.array_equiv(arr1, lcs_res["res"]):
                        find["res"] = True

                #print("confronting")
                for arr2 in self.log_templ["req"]:
                    if np.array_equiv(arr2, lcs_res["req"]):
                 #       print(arr, lcs_res["req"], len(self.log_templ["req"]))
                        find["req"] = True

                if not find["err"]:
                    self.log_templ["err"].append(lcs_res["err"])
                if not find["res"]:
                    self.log_templ["res"].append(lcs_res["res"])
                if not find["req"]:
                    self.log_templ["req"].append(lcs_res["err"])
        
    def dump_results(self):
        print("== Printing Content vocabulary ERR: ==")
        voc = 0
        for k in self.log_templ["err"]:
            print(f"{voc}: {k}")
            voc += 1
        print("== Vocabulary Ended ==")
        print("== Printing Content vocabulary RES: ==")
        voc = 0
        for k in self.log_templ["res"]:
            print(f"{voc}: {k}")
            voc += 1
        print("== Vocabulary Ended ==")
        print("== Printing Content vocabulary REQ: ==")
        voc = 0
        for k in self.log_templ["req"]:
            print(f"{voc}: {k}")
            voc += 1
        print("== Vocabulary Ended ==")

    def parse(self, inlog):
        log_1 = dict()
        log_1["err"] = inlog.get("errorMessage")      if not None else "None"
        log_1["res"] = inlog.get("responseElements")  if not None else "None"
        log_1["req"] = inlog.get("requestParameters") if not None else "None"
        
        longest = dict()    # the array with most common chars
        longest["err"], longest["res"], longest["req"] = [], [], []
        max_len = dict()    # the max len of the match
        max_len["err"], max_len["res"], max_len["req"] = 0, 0, 0
        pos = dict()        # type of the log based on the position
        pos["err"], pos["res"], pos["req"] = 0, 0, 0

        index = 0
        for e1 in self.log_templ["err"]:
            lcs_res = self.LCS(log_1["err"], "".join(e1))
            if len(lcs_res) >= max_len["err"]:
                max_len["err"] = len(lcs_res)
                longest["err"] = lcs_res
                pos["err"] = index 
            index += 1
        index = 0
        for e1 in self.log_templ["res"]:
            lcs_res = self.LCS(log_1["res"], "".join(e1))
            if len(lcs_res) >= max_len["res"]:
                max_len["res"] = len(lcs_res)
                longest["res"] = lcs_res
                pos["res"] = index
            index += 1
        index = 0
        for e1 in self.log_templ["req"]:
            lcs_res = self.LCS(log_1["req"], "".join(e1))
            if len(lcs_res) >= max_len["req"]:
                max_len["req"] = len(lcs_res)
                longest["req"] = lcs_res
                pos["req"] = index
            index += 1
        
        print(f"Position Err: {pos['err']}, Res: {pos['res']}, Req: {pos['req']}")


if __name__ == "__main__":

    flaws_cloudtrail00  = "../../data/raw/"
    test_filename       = "../data/raw/flaws_cloudtrail00.json"
    test_input          = "../input.json"

    #indir       = "./"
    filename    = "flaws_cloudtrail00.json"
    outdir      = "../data/results/sputo/"
    #log_format  = None, 
    # tau=0.5, 
    # rex=[], 
    # keep_para=True
    spell = SputuLogParser(
        indir=flaws_cloudtrail00, 
        outdir=outdir, 
        filename=filename
    )
    spell.create_vocabulary(until=50)
    spell.dump_results()

    # print("\n ==== Testing ==== \n")
    # with open(flaws_cloudtrail00, "r") as fj:
    #     j_obj = json.loads(fj.read())
    #     for rec in j_obj["Records"]:
    #         spell.parse(rec)%  